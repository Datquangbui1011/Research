"""The six topological approaches on OpenOOD's ResNet-18 (per-image scores).

ResNet-18 has a single dense layer, fc: 512 penultimate features h (non-negative, after ReLU and
global average pooling) -> K classes. Its activation graph is the bipartite graph with edge weights
|W(k,i) h(i)| ([2]), so the graph-based methods [1]-[4] use this one layer:
  [1] NP of the activation graph, z-scored per predicted class (input-dependent adaptation)
  [2] TU: 2-Wasserstein distance of the 0-dim PD (MST weights) to the class-mean training PD
  [3] pTU: Frechet variance of the PDs under 30 samples of a last-layer Laplace posterior
  [4] persistent Laplacian (= graph Laplacian of the thresholded graph in dim 0), z-scored per class
[5] local ΔH0 uses point clouds of every stage (spatial means of conv1, layer1-3, pooled layer4).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import bipartite_mst_weights  # noqa: E402

from bench import BENCHMARKS, CACHE, features, fc_weights, subsample_idx  # noqa: E402
from evaluate import test_splits  # noqa: E402

N_TRAIN_REF = 10000  # training images used for class-wise reference statistics
S = 30               # posterior samples for pTU
N_PL = 1000          # subsample per test set for the persistent Laplacian


def _train_ref(bench, seed):
    tr = features(bench, seed, f"train_{bench}")
    idx = np.random.default_rng(5).choice(len(tr["feat"]), N_TRAIN_REF, replace=False)
    return tr["feat"][idx].astype(np.float32), tr["logits"][idx]


def _pds(W, H, chunk=2000):
    """0-dim PDs (sorted MST weights) of the fc activation graphs |W(k,i) h(i)| for rows of H."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"  # ~3x faster than CPU for these graphs
    Wt = torch.from_numpy(np.abs(W).T.astype(np.float32)).to(dev)  # (512, K)
    out = []
    for c in range(0, len(H), chunk):
        h = torch.from_numpy(np.abs(H[c:c + chunk]).astype(np.float32)).to(dev)
        out.append(bipartite_mst_weights(h[:, :, None] * Wt[None]).cpu())
    return torch.cat(out).numpy()


def _cached(name, bench, seed, fn):
    f = CACHE / f"{name}_{bench}_s{seed}.npz"
    if f.exists():
        return dict(np.load(f))
    out = fn()
    np.savez(f, **out)
    return out


def _test_pds(bench, seed):
    W, _ = fc_weights(bench, seed)
    def build():
        out = {s: _pds(W, features(bench, seed, s)["feat"]) for s in test_splits(bench)}
        h, lg = _train_ref(bench, seed)
        out["train"] = _pds(W, h)
        return out
    return _cached("pd", bench, seed, build)


def _pred(bench, seed, split):
    return features(bench, seed, split)["logits"].argmax(1)


# --------------------------------------------------------------------------- [1] neural persistence
def _np(pd, min_edge):
    """Normalized NP of one layer (Rieck et al.): weights / max, persistence 1 - w', p = 2."""
    w = pd / pd[:, :1]
    return np.sqrt(((1 - w) ** 2).sum(1)) / ((1 - min_edge / pd[:, 0]) * np.sqrt(pd.shape[1]))


def neural_persistence(bench, seed):
    pds = _test_pds(bench, seed)
    W, _ = fc_weights(bench, seed)
    wmin = np.abs(W).min(0)  # smallest |W(k,i)| per input neuron

    def npv(split, feat):
        return _np(pds[split], (np.abs(feat) * wmin).min(1))
    htr, lgtr = _train_ref(bench, seed)
    vtr, ktr = npv("train", htr), lgtr.argmax(1)
    K = BENCHMARKS[bench]["classes"]
    mu = np.array([vtr[ktr == k].mean() for k in range(K)])
    sd = np.array([vtr[ktr == k].std() + 1e-12 for k in range(K)])
    out = {}
    for s in test_splits(bench):
        k = _pred(bench, seed, s)
        out[s] = np.abs(npv(s, features(bench, seed, s)["feat"]) - mu[k]) / sd[k]
    return out


# --------------------------------------------------------------------------- [2] topological uncertainty
def topological_uncertainty(bench, seed):
    pds = _test_pds(bench, seed)
    _, lgtr = _train_ref(bench, seed)
    ktr = lgtr.argmax(1)
    K = BENCHMARKS[bench]["classes"]
    means = np.stack([pds["train"][ktr == k].mean(0) for k in range(K)])
    return {s: np.linalg.norm(pds[s] - means[_pred(bench, seed, s)], axis=1) for s in test_splits(bench)}


# --------------------------------------------------------------------------- [3] pTU with a last-layer Laplace posterior
def laplace_posterior(bench, seed):
    """Diagonal last-layer Laplace approximation (generalized Gauss-Newton) for fc (W, b), with the
    prior precision chosen by the validation log-likelihood of the 30-sample predictive."""
    W, b = fc_weights(bench, seed)
    tr = features(bench, seed, f"train_{bench}")
    h = tr["feat"].astype(np.float64)
    p = np.exp(tr["logits"] - logsumexp(tr["logits"], 1, keepdims=True))
    g = p * (1 - p)                                   # diag of the softmax Hessian, per class
    ggn_w = g.T @ (h ** 2)                            # (K, 512)
    ggn_b = g.sum(0)
    val = features(bench, seed, f"val_{bench}")
    hv, yv = val["feat"].astype(np.float64), val["y"]
    rng = np.random.default_rng(seed)
    best = None
    for tau in [1e-1, 1, 10, 100, 1000, 1e4]:
        sw, sb = 1 / np.sqrt(ggn_w + tau), 1 / np.sqrt(ggn_b + tau)
        probs = 0
        for _ in range(S):
            lg = hv @ (W + sw * rng.standard_normal(W.shape)).T + b + sb * rng.standard_normal(b.shape)
            probs = probs + np.exp(lg - logsumexp(lg, 1, keepdims=True)) / S
        ll = np.log(probs[np.arange(len(yv)), yv] + 1e-12).mean()
        if best is None or ll > best[0]:
            best = (ll, tau)
    tau = best[1]
    return W, b, 1 / np.sqrt(ggn_w + tau), 1 / np.sqrt(ggn_b + tau), tau


def _ptu_all(bench, seed):
    def build():
        W, b, sw, sb, tau = laplace_posterior(bench, seed)
        rng = np.random.default_rng(100 + seed)
        samples = [(W + sw * rng.standard_normal(W.shape), b + sb * rng.standard_normal(b.shape)) for _ in range(S)]
        out = {"tau": np.array(tau)}
        splits = test_splits(bench) + ["train"]
        for s in splits:
            if s == "train":
                h = _train_ref(bench, seed)[0][:2000]
            else:  # fixed N_SUB subsample per test set (30 posterior samples per image)
                f = features(bench, seed, s)["feat"]
                h = f[subsample_idx(len(f))].astype(np.float32)
            pds, probs, ent_each = [], 0, 0
            for Ws, bs in samples:
                pds.append(_pds(Ws, h))
                lg = h @ Ws.T + bs
                pr = np.exp(lg - logsumexp(lg, 1, keepdims=True))
                probs = probs + pr / S
                ent_each = ent_each - (pr * np.log(pr + 1e-12)).sum(1) / S
            P = np.stack(pds)
            out[f"ptu_{s}"] = ((P - P.mean(0)) ** 2).sum(2).mean(0)
            out[f"mi_{s}"] = -(probs * np.log(probs + 1e-12)).sum(1) - ent_each
        return out
    return _cached("ptu", bench, seed, build)


def ptu(bench, seed):
    d = _ptu_all(bench, seed)
    return {s: d[f"ptu_{s}"] for s in test_splits(bench)}


def ptu_two_sided(bench, seed):
    """Our variant: |log pTU - median log pTU(train)|, so unusually low variance is also flagged."""
    d = _ptu_all(bench, seed)
    ref = np.median(np.log(d["ptu_train"] + 1e-12))
    return {s: np.abs(np.log(d[f"ptu_{s}"] + 1e-12) - ref) for s in test_splits(bench)}


def mutual_information(bench, seed):
    d = _ptu_all(bench, seed)
    return {s: d[f"mi_{s}"] for s in test_splits(bench)}


# --------------------------------------------------------------------------- [4] persistent Laplacian
PL_FEATURES = ["beta0", "edges", "lambda_min+", "lambda_max", "lambda_std"]


def _laplacian_features(G, thresholds, tol=1e-6):
    out = []
    for t in thresholds:
        A = G >= t
        ki, kj = A.any(1), A.any(0)
        iso = int((~ki).sum() + (~kj).sum())
        A = A[ki][:, kj].astype(np.float64)
        E = A.sum()
        if E == 0:
            out.append([iso, 0, 0, 0, 0]); continue
        a, c = A.shape
        L = np.zeros((a + c, a + c))
        L[:a, a:], L[a:, :a] = -A, -A.T
        L += np.diag(-L.sum(1))
        ev = np.linalg.eigvalsh(L)
        nz = ev[ev > tol * max(1.0, ev[-1])]
        out.append([iso + a + c - len(nz), E, nz[0], ev[-1], nz.std() if len(nz) > 1 else 0])
    return out


def _pl_all(bench, seed):
    def build():
        W, _ = fc_weights(bench, seed)
        Wt = np.abs(W).T
        pds = _test_pds(bench, seed)
        tr_pd = pds["train"]
        thresholds = np.quantile(tr_pd[tr_pd > 0], (0.5, 0.75, 0.9))
        out = {"thresholds": thresholds}
        htr, lgtr = _train_ref(bench, seed)
        items = [("train", htr[:2000], lgtr[:2000])]
        for s in test_splits(bench):
            d = features(bench, seed, s)
            idx = subsample_idx(len(d["feat"]))[:N_PL]
            items.append((s, d["feat"][idx], d["logits"][idx]))
        for s, h, lg in items:
            out[f"f_{s}"] = np.array([_laplacian_features(np.abs(hh)[:, None] * Wt, thresholds) for hh in h.astype(np.float64)])
            out[f"k_{s}"] = lg.argmax(1)
        return out
    return _cached("pl", bench, seed, build)


def _pl_score(bench, seed, cols):
    d = _pl_all(bench, seed)
    K = BENCHMARKS[bench]["classes"]
    ftr, ktr = d["f_train"], d["k_train"]
    gmean = ftr.mean(0)
    mu = np.stack([ftr[ktr == k].mean(0) if (ktr == k).any() else gmean for k in range(K)])
    sd = ftr.std(0) + 1e-9
    idx = [PL_FEATURES.index(c) for c in cols]
    return {s: (((d[f"f_{s}"] - mu[d[f"k_{s}"]]) / sd)[..., idx] ** 2).mean((1, 2)) for s in test_splits(bench)}


def pl_full(bench, seed):
    return _pl_score(bench, seed, PL_FEATURES)


def pl_beta0(bench, seed):
    return _pl_score(bench, seed, ["beta0"])


# --------------------------------------------------------------------------- [5] local ΔH0 on stage point clouds
def _local_all(bench, seed):
    from m5_data_manifold import local_delta

    def build():
        tr = features(bench, seed, f"train_{bench}", stages=True)
        ref = [x[:5000].astype(np.float64) for x in tr["stage_mean"]]
        queries = {"train": [x[5000:6000].astype(np.float64) for x in tr["stage_mean"]]}
        for s in test_splits(bench):
            d = features(bench, seed, s, stages=True)
            queries[s] = [x.astype(np.float64) for x in d["stage_mean"]]
        return {f"{s}": np.stack([local_delta(ref[l], q[l]) for l in range(len(ref))], 1) for s, q in queries.items()}
    return _cached("local", bench, seed, build)


def local_dh0(bench, seed):
    d = _local_all(bench, seed)
    tr = d["train"][..., 0]
    mu, sd = tr.mean(0), tr.std(0) + 1e-12
    return {s: (((d[s][..., 0] - mu) / sd) ** 2).mean(1) for s in test_splits(bench)}


# name -> (function, evaluated on the subsample?)
TOPO_METHODS = {
    "[1] NP": (neural_persistence, False),
    "[2] TU": (topological_uncertainty, False),
    "[3] pTU": (ptu, True),
    "[3] pTU two-sided": (ptu_two_sided, True),
    "MI (Laplace)": (mutual_information, True),
    "[4] PL full": (pl_full, True),
    "[4] PL beta0": (pl_beta0, True),
    "[5] local dH0": (local_dh0, True),
}
