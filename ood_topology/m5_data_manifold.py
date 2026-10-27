"""Method [5]: topology and geometry of the data manifold across layers (Magai & Ayzenberg, 2022)
for OOD detection.

Each layer l gives a point cloud X_l = {h_l(x_1), ..., h_l(x_n)} (one point per image; layers:
input 784, hidden 64, hidden 16, logits 10). On X_l we compute Vietoris-Rips persistent
homology in dimensions 0 and 1 and summarize it, as in the paper, by the total persistence
(sum of lifetimes) of H0 and H1, plus the intrinsic dimension (TwoNN estimator).

(a) Batch level (the paper's setting: descriptors of a set of points). A batch of m test images
    (m = 10, 20, 50) gives a 12-dim descriptor (4 layers x {H0, H1, ID}). Score = mean squared
    z-score against descriptors of batches of MNIST training images (5000 images cut into
    batches of the same size). AUROC over 2000/m ID test batches vs. 2000/m batches of each OOD set. TU [2] and MSP batch means are evaluated on the same batches.
(b) Per sample (our local adaptation). For input x and layer l, P = its k = 20 nearest training
    points in layer l. Features = change in H0 and H1 total persistence of Rips(P) when x is
    added. Score = mean squared z-score w.r.t. training inputs. Compared with the plain k-NN
    distance (Sun et al. 2022), since the H0 change is closely related to it.
"""
import argparse
import json

import gudhi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist, squareform

from common import (ID, CACHE_DIR, RESULTS_DIR, eval_sets, forward_trace, load_dataset, load_model,
                    mean_net_pds, msp_score, ood_metrics)
from m2_topological_uncertainty import class_mean_pds, tu_score

LAYER_NAMES = ["input", "hidden1", "hidden2", "logits"]
BATCH_SIZES = (10, 20, 50)
N_REF_IMAGES = 5000  # reference batches are cut from 5000 training images
N_LOCAL = 1000
K_NN = 20


@torch.no_grad()
def layer_clouds(weights, X):
    inputs, logits = forward_trace(weights, X)
    return [h.numpy().astype(np.float64) for h in inputs] + [logits.numpy().astype(np.float64)]


def rips_total_persistence(D):
    """Total persistence (sum of finite lifetimes) of H0 and H1 of the Rips filtration."""
    st = gudhi.RipsComplex(distance_matrix=D).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    out = []
    for dim in (0, 1):
        iv = st.persistence_intervals_in_dimension(dim)
        iv = iv[np.isfinite(iv[:, 1])] if len(iv) else iv
        out.append(float((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0.0)
    return out


def twonn_dimension(D):
    """TwoNN intrinsic-dimension MLE (Facco et al. 2017) from a distance matrix."""
    Ds = np.sort(D + np.diag(np.full(len(D), np.inf)), 1)
    r1, r2 = Ds[:, 0], Ds[:, 1]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    return float(ok.sum() / np.log(mu).sum()) if ok.sum() > 1 else 0.0


def batch_descriptor(clouds):
    feats = []
    for P in clouds:
        D = squareform(pdist(P))
        feats += rips_total_persistence(D) + [twonn_dimension(D)]
    return feats


def batches_of(clouds, m):
    return np.array([batch_descriptor([c[b * m:(b + 1) * m] for c in clouds]) for b in range(len(clouds[0]) // m)])


def local_delta(ref, queries, k=K_NN, skip_self=False):
    """Change of H0/H1 total persistence of the k-NN cloud when the query point is added,
    plus the k-th NN distance (k-NN baseline)."""
    d = cdist(queries, ref)
    nn = np.argsort(d, 1)[:, 1:k + 1] if skip_self else np.argsort(d, 1)[:, :k]
    out = np.zeros((len(queries), 3))
    for i, q in enumerate(queries):
        P = ref[nn[i]]
        D0 = squareform(pdist(P))
        D1 = squareform(pdist(np.vstack([P, q[None]])))
        h_without, h_with = rips_total_persistence(D0), rips_total_persistence(D1)
        out[i] = [h_with[0] - h_without[0], h_with[1] - h_without[1], d[i, nn[i, -1]]]
    return out


def zscore_score(ref, x):
    mu, sd = ref.mean(0), ref.std(0) + 1e-12
    return (((x - mu) / sd) ** 2).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    rng = np.random.default_rng(seed)

    cache = CACHE_DIR / f"pd_manifold_s{seed}.pt"
    if cache.exists():
        res = torch.load(cache, weights_only=False)
    else:
        weights = load_model(seed).sample_weights(mean=True)
        Xtr, _ = load_dataset(ID, "train")
        perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(21))
        ref_batches_X = Xtr[perm[:N_REF_IMAGES]]              # 5000 images: reference batches
        cloud_X = Xtr[perm[5000:10000]]                      # 5000 images: k-NN reference cloud
        query_tr_X = Xtr[perm[10000:10000 + N_LOCAL]]         # 1000 images: train scores (local)
        ref_clouds = layer_clouds(weights, ref_batches_X)
        res = {"ref_batches": {m: batches_of(ref_clouds, m) for m in BATCH_SIZES}}
        print("  reference batch descriptors done", flush=True)
        knn_clouds = layer_clouds(weights, cloud_X)
        q_tr = layer_clouds(weights, query_tr_X)
        res["local_train"] = np.stack([local_delta(knn_clouds[l], q_tr[l]) for l in range(4)], 1)
        for name in eval_sets():
            X, _ = load_dataset(name, "test", n=2000)
            cl = layer_clouds(weights, X)
            batches = {m: batches_of(cl, m) for m in BATCH_SIZES}
            local = np.stack([local_delta(knn_clouds[l], cl[l][:N_LOCAL]) for l in range(4)], 1)
            res[name] = {"batches": batches, "local": local}
            print(f"  computed descriptors for {name}", flush=True)
        torch.save(res, cache)

    # ---------------- (a) batch level
    tr_pd = mean_net_pds(seed, ID, "train", 10000)
    means = class_mean_pds(tr_pd["pds"], tr_pd["logits"].argmax(1))
    tu_all, msp_all = {}, {}
    for n in eval_sets():
        d = mean_net_pds(seed, n, "test", 2000)
        tu_all[n] = tu_score(d["pds"], d["logits"], means)[0]
        msp_all[n] = msp_score(d["logits"])

    def batch_means(v, m):
        return v[:len(v) // m * m].reshape(-1, m).mean(1)

    feat_names = [f"{l}:{f}" for l in LAYER_NAMES for f in ("H0", "H1", "ID")]
    batch_results = {}
    for m in BATCH_SIZES:
        ref = res["ref_batches"][m]
        sc = {"manifold": {n: zscore_score(ref, res[n]["batches"][m]) for n in eval_sets()},
              "tu": {n: batch_means(tu_all[n], m) for n in eval_sets()},
              "msp": {n: batch_means(msp_all[n], m) for n in eval_sets()}}
        rows = []
        print(f"\n(a) batch level, m = {m} images/batch, {2000 // m} ID vs {2000 // m} OOD batches (AUROC / FPR95)")
        print(f"{'OOD set':14s} | {'[5] descriptors':>15s} | {'TU batch mean':>14s} | {'MSP batch mean':>14s}")
        for n in eval_sets()[1:]:
            r = {"ood": n}
            for key in sc:
                a, f = ood_metrics(sc[key][ID], sc[key][n])
                r[key] = {"auroc": a, "fpr95": f}
            rows.append(r)
            print(f"{n:14s} | " + " | ".join(f"{r[k]['auroc']:6.1f} / {r[k]['fpr95']:5.1f}" for k in sc))
        avg_b = {k: {q: float(np.mean([r[k][q] for r in rows])) for q in ("auroc", "fpr95")} for k in sc}
        print(f"{'average':14s} | " + " | ".join(f"{avg_b[k]['auroc']:6.1f} / {avg_b[k]['fpr95']:5.1f}" for k in sc))
        batch_results[m] = {"rows": rows, "average": avg_b}

    ref = res["ref_batches"][50]
    print("\nmean batch descriptors, m = 50 (H0 / H1 total persistence, TwoNN ID) per layer:")
    print(f"  {'set':14s}" + "".join(f"{l:>24s}" for l in LAYER_NAMES))
    desc_means = {}
    for n in ["reference (train)"] + eval_sets():
        v = (ref if n.startswith("reference") else res[n]["batches"][50]).mean(0)
        desc_means[n] = v.tolist()
        print(f"  {n:14s}" + "".join(f"{v[3 * l]:9.1f}/{v[3 * l + 1]:6.2f}/{v[3 * l + 2]:5.1f}" for l in range(4)))

    # ---------------- (b) per sample, local
    lt = res["local_train"]  # (N, 4 layers, 3)
    loc_rows = []
    variants = {"local PH (dH0, dH1)": [0, 1], "local dH0 only": [0], "local dH1 only": [1], "kNN distance": [2]}
    print(f"\n(b) per sample, k = {K_NN} nearest training points per layer, first 1000 images (AUROC / FPR95)")
    print(f"{'OOD set':14s} | " + " | ".join(f"{v:>20s}" for v in variants))
    sc = {v: {n: zscore_score(lt[..., idx].reshape(len(lt), -1), res[n]["local"][..., idx].reshape(N_LOCAL, -1))
              for n in eval_sets()} for v, idx in variants.items()}
    for n in eval_sets()[1:]:
        r = {"ood": n}
        line = f"{n:14s} | "
        for v in variants:
            a, f = ood_metrics(sc[v][ID], sc[v][n])
            r[v] = {"auroc": a, "fpr95": f}
            line += f"{'':7s}{a:5.1f} / {f:5.1f} | "
        loc_rows.append(r)
        print(line)
    avg_l = {v: {m: float(np.mean([r[v][m] for r in loc_rows])) for m in ("auroc", "fpr95")} for v in variants}
    print(f"{'average':14s} | " + " | ".join(f"{'':7s}{avg_l[v]['auroc']:5.1f} / {avg_l[v]['fpr95']:5.1f}" for v in variants))

    (RESULTS_DIR / f"m5_data_manifold_s{seed}.json").write_text(json.dumps(
        {"seed": seed, "k": K_NN, "batch": {str(m): v for m, v in batch_results.items()},
         "descriptor_means": desc_means, "descriptor_names": feat_names,
         "local_rows": loc_rows, "local_average": avg_l}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, j, title in zip(axes, range(3), ["H0 total persistence", "H1 total persistence", "intrinsic dim. (TwoNN)"]):
        for n in ["reference (train)"] + eval_sets():
            v = np.array(desc_means[n])[j::3]
            v = v / np.array(desc_means["reference (train)"])[j::3]
            ax.plot(LAYER_NAMES, v, marker="o", lw=2.2 if n in (ID, "reference (train)") else 1,
                    ls="--" if n.startswith("reference") else "-", label=n)
        ax.set_title(f"[5] {title}, m = 50 (rel. to {ID} train)", fontsize=10)
        ax.set_yscale("log")
    axes[0].set_ylabel("batch mean / train mean")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"m5_data_manifold_s{seed}.png", dpi=150)


if __name__ == "__main__":
    main()
