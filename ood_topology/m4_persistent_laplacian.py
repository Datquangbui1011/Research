"""Method [4]: persistent Laplacian (Wolf et al. 2025; computation as in PETLS / Botnan-Dong)
on the activation graphs of [2], for OOD detection.

Reduction used here. Each layer's activation graph |W(j,i) x(i)| is filtered by superlevel
sets: K_t = all vertices + the edges with weight >= t. Because every vertex is present from the
start, for s >= t (K_s ⊆ K_t) the 0-dim persistent Laplacian is
    Delta_0^{K_s, K_t} = Delta_0,up^{K_s, K_t} (+ Delta_0,down = 0) = L(K_t),
the combinatorial graph Laplacian of K_t. A bipartite graph has no triangles, so dim 0 is the
only non-trivial degree (as the report notes). Its spectrum gives:
  - the number of zero eigenvalues = beta_0(K_t), the persistent Betti number, which is the
    same information as the 0-dim PD used in [1]-[3];
  - the nonzero eigenvalues, the extra geometric information the persistent Laplacian adds.

Features per layer and threshold t (t = fixed quantiles of the nonzero training MST weights
of that layer): beta_0, number of edges (= trace / 2), smallest nonzero eigenvalue, largest eigenvalue,
and the std of the nonzero eigenvalues. OOD score = mean squared z-score of the features with
respect to training inputs of the same predicted class (the same class-conditional idea as TU).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (ID, CACHE_DIR, RESULTS_DIR, activation_graphs, eval_sets, load_dataset, load_model,
                    mean_net_pds, msp_score, ood_metrics)
from m2_topological_uncertainty import class_mean_pds, tu_score

N_EVAL_PL = 1000
N_TRAIN_PL = 2000
QUANTILES = (0.5, 0.75, 0.9)
FEATURES = ["beta0", "edges", "lambda_min+", "lambda_max", "lambda_std"]
PH_FEATURES = ["beta0"]  # what the PD already contains
SPEC_FEATURES = ["edges", "lambda_min+", "lambda_max", "lambda_std"]  # the extra PL information


def laplacian_features(G, t, tol=1e-6):
    """Spectral features of L(K_t) for one bipartite weight matrix G (n, m)."""
    A = G >= t
    n, m = A.shape
    keep_i, keep_j = A.any(1), A.any(0)
    isolated = int((~keep_i).sum() + (~keep_j).sum())
    A = A[keep_i][:, keep_j].double()
    E = float(A.sum())
    if E == 0:
        return [isolated, 0.0, 0.0, 0.0, 0.0]
    a, b = A.shape
    L = torch.zeros(a + b, a + b, dtype=torch.float64)
    L[:a, a:] = -A
    L[a:, :a] = -A.t()
    L += torch.diag(-L.sum(1))
    ev = torch.linalg.eigvalsh(L)
    nz = ev[ev > tol * max(1.0, float(ev[-1]))]
    beta0 = isolated + (a + b - len(nz))
    return [beta0, E, float(nz[0]), float(ev[-1]), float(nz.std()) if len(nz) > 1 else 0.0]


@torch.no_grad()
def pl_features(weights, X, thresholds):
    graphs, _, logits = activation_graphs(weights, X)
    feats = np.zeros((len(X), len(graphs), len(thresholds[0]), len(FEATURES)))
    for b in range(len(X)):
        for l, G in enumerate(graphs):
            for q, t in enumerate(thresholds[l]):
                feats[b, l, q] = laplacian_features(G[b], t)
    return feats, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed

    cache = CACHE_DIR / f"pd_pl_s{seed}.pt"
    tr_pd = mean_net_pds(seed, ID, "train", 10000)
    # quantiles of the nonzero training MST weights (most layer-1 weights are 0: black pixels)
    thresholds = [np.quantile(pd.numpy()[pd.numpy() > 0], QUANTILES) for pd in tr_pd["pds"]]
    if cache.exists():
        res = torch.load(cache, weights_only=False)
    else:
        model = load_model(seed)
        weights = model.sample_weights(mean=True)
        res = {}
        Xtr, _ = load_dataset(ID, "train", n=N_TRAIN_PL, seed=11)
        sets = [("train", Xtr)] + [(n, load_dataset(n, "test", n=2000)[0][:N_EVAL_PL]) for n in eval_sets()]
        for name, X in sets:
            f, lg = pl_features(weights, X, thresholds)
            res[name] = {"feats": f, "logits": lg}
            print(f"  computed PL spectra for {name}", flush=True)
        torch.save(res, cache)

    # class-conditional z-scores (global std per feature)
    tr = res["train"]
    pred_tr = tr["logits"].argmax(1).numpy()
    mu = np.stack([tr["feats"][pred_tr == k].mean(0) for k in range(10)])
    sd = tr["feats"].std(0) + 1e-9

    def score(r, feat_names):
        idx = [FEATURES.index(f) for f in feat_names]
        z = (r["feats"] - mu[r["logits"].argmax(1).numpy()]) / sd
        return (z[..., idx] ** 2).mean((1, 2, 3)), (z[..., idx] ** 2).mean((2, 3))

    variants = {"PL full": FEATURES, "beta0 only (PH info)": PH_FEATURES, "nonzero spectrum only": SPEC_FEATURES}
    scores = {v: {n: score(r, f) for n, r in res.items()} for v, f in variants.items()}

    # TU [2] and MSP on the same 1000-image subsets, for a fair comparison
    means = class_mean_pds(tr_pd["pds"], tr_pd["logits"].argmax(1))
    tu, msp = {}, {}
    for n in eval_sets():
        d = mean_net_pds(seed, n, "test", 2000)
        pds = [p[:N_EVAL_PL] for p in d["pds"]]
        tu[n] = tu_score(pds, d["logits"][:N_EVAL_PL], means)[0]
        msp[n] = msp_score(d["logits"][:N_EVAL_PL])

    names = list(variants) + ["TU [2]", "MSP"]
    rows = []
    print(f"\nthresholds (MST-weight quantiles {QUANTILES}) per layer: "
          + "; ".join(np.round(t, 3).astype(str).tolist().__str__() for t in thresholds))
    print(f"{'OOD set':14s} | " + " | ".join(f"{v[:21]:>21s}" for v in names) + " | PL-full per-layer AUROC")
    for name in eval_sets()[1:]:
        r = {"ood": name}
        line = f"{name:14s} | "
        for v in names:
            if v in variants:
                s_id, s_ood = scores[v][ID][0], scores[v][name][0]
            else:
                s_id, s_ood = (tu[ID], tu[name]) if v.startswith("TU") else (msp[ID], msp[name])
            a, f = ood_metrics(s_id, s_ood)
            r[v] = {"auroc": a, "fpr95": f}
            line += f"{'':9s}{a:5.1f} / {f:5.1f} | "
        r["layer_auroc"] = [ood_metrics(scores["PL full"][ID][1][:, l], scores["PL full"][name][1][:, l])[0]
                            for l in range(3)]
        print(line + " ".join(f"{x:5.1f}" for x in r["layer_auroc"]))
        rows.append(r)
    avg = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in ("auroc", "fpr95")} for v in names}
    avg["layer_auroc"] = np.mean([r["layer_auroc"] for r in rows], 0).tolist()
    print(f"{'average':14s} | " + " | ".join(f"{'':9s}{avg[v]['auroc']:5.1f} / {avg[v]['fpr95']:5.1f}" for v in names)
          + " | " + " ".join(f"{x:5.1f}" for x in avg["layer_auroc"]))

    # mean feature values (layer 1, middle threshold) to see what changes
    print("\nmean features, layer 1, threshold q=0.75:  " + "  ".join(f"{f:>11s}" for f in FEATURES))
    for n in ["train"] + eval_sets():
        print(f"  {n:14s}" + "  ".join(f"{x:11.3f}" for x in res[n]["feats"][:, 0, 1].mean(0)))

    (RESULTS_DIR / f"m4_persistent_laplacian_s{seed}.json").write_text(json.dumps(
        {"seed": seed, "quantiles": QUANTILES, "thresholds": [t.tolist() for t in thresholds],
         "n_eval": N_EVAL_PL, "rows": rows, "average": avg}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, f in zip(axes, ["beta0", "lambda_max", "lambda_min+"]):
        i = FEATURES.index(f)
        vals = {n: res[n]["feats"][:, 0, 1, i] for n in eval_sets()}
        lo, hi = np.quantile(np.concatenate(list(vals.values())), [0.005, 0.995])
        bins = np.linspace(lo, hi, 50)
        for n, v in vals.items():
            ax.hist(v, bins=bins, histtype="step", density=True, lw=2.2 if n == ID else 1,
                    label=n + (" (ID)" if n == ID else ""))
        ax.set_xlabel(f"{f} of L(K_t), layer 1, t = q0.75")
        ax.set_ylabel("density")
    axes[0].set_title("[4] zero eigenvalues = beta_0 (PH part)")
    axes[1].set_title("largest eigenvalue (extra PL info)")
    axes[2].set_title("smallest nonzero eigenvalue (extra PL info)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"m4_persistent_laplacian_s{seed}.png", dpi=150)


if __name__ == "__main__":
    main()
