"""Method [6]: topology of hidden-layer activations compared by sliced Wasserstein distance
(Purvine et al., AAAI 2023), for OOD detection.

Purvine et al. build a point cloud per layer from the activations of a set of images, compute
its persistence diagrams, and compare diagrams with the sliced Wasserstein (SW) distance. For a
CNN they keep one c-dim vector per image (at the spatial position with the largest
activation). For our MLP every image already gives one vector per layer, so that step is not
needed. Their Mapper analysis is qualitative and is not used here.

Everything is batch level (diagrams describe a set of images). Batches of m test images,
m = 10, 20, 50; layers: input, hidden1, hidden2, logits; Rips PDs in dimensions 0 and 1.
  (i)  "SW to reference": for each layer and dimension, the mean SW distance between the batch
       PD and the PDs of 20 anchor batches of MNIST training images (8 numbers). Score = mean
       z-score of these distances against calibration batches of training images (larger
       distance = more OOD).
  (ii) "layer profile": SW distances between the PDs of consecutive layers inside the batch,
       Purvine's layer-similarity measure (3 layer pairs x 2 dims = 6 numbers). Score = mean
       squared z-score against the training-batch profiles.
"""
import argparse
import json

import gudhi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from common import ID, CACHE_DIR, RESULTS_DIR, eval_sets, load_dataset, load_model, ood_metrics
from m5_data_manifold import LAYER_NAMES, layer_clouds

BATCH_SIZES = (10, 20, 50)
N_ANCHORS = 20
N_DIRECTIONS = 50
THETAS = np.linspace(-np.pi / 2, np.pi / 2, N_DIRECTIONS, endpoint=False)
DIRS = np.stack([np.cos(THETAS), np.sin(THETAS)], 1)


def rips_diagrams(P):
    """Finite H0 and H1 diagrams (arrays of (birth, death)) of the Rips filtration of P."""
    st = gudhi.RipsComplex(distance_matrix=squareform(pdist(P))).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    out = []
    for dim in (0, 1):
        iv = st.persistence_intervals_in_dimension(dim)
        iv = iv[np.isfinite(iv[:, 1])] if len(iv) else np.zeros((0, 2))
        out.append(iv.reshape(-1, 2))
    return out


def sliced_wasserstein(D1, D2):
    """Sliced Wasserstein distance between two persistence diagrams (Carriere et al. 2017):
    each diagram is completed with the diagonal projections of the other's points, both are
    projected on N_DIRECTIONS lines, and the sorted 1-D projections are compared in L1."""
    def diag(D):
        m = D.mean(1, keepdims=True)
        return np.hstack([m, m])
    A = np.vstack([D1, diag(D2)])
    B = np.vstack([D2, diag(D1)])
    if len(A) == 0:
        return 0.0
    pa, pb = np.sort(A @ DIRS.T, 0), np.sort(B @ DIRS.T, 0)
    return float(np.abs(pa - pb).sum(0).mean())


def batch_diagrams(clouds, m):
    """List over batches of [layer][dim] diagrams."""
    return [[rips_diagrams(c[b * m:(b + 1) * m]) for c in clouds] for b in range(len(clouds[0]) // m)]


def sw_to_anchors(batch, anchors):
    return [np.mean([sliced_wasserstein(batch[l][d], a[l][d]) for a in anchors])
            for l in range(len(batch)) for d in (0, 1)]


def layer_profile(batch):
    return [sliced_wasserstein(batch[l][d], batch[l + 1][d]) for l in range(len(batch) - 1) for d in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed

    cache = CACHE_DIR / f"pd_swph_s{seed}.pt"
    if cache.exists():
        res = torch.load(cache, weights_only=False)
    else:
        weights = load_model(seed).sample_weights(mean=True)
        Xtr, _ = load_dataset(ID, "train")
        perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(21))
        ref_clouds = layer_clouds(weights, Xtr[perm[:5000]])
        res = {}
        for m in BATCH_SIZES:
            ref = batch_diagrams(ref_clouds, m)
            anchors, calib = ref[:N_ANCHORS], ref[N_ANCHORS:]
            res[m] = {"calib": {"sw": np.array([sw_to_anchors(b, anchors) for b in calib]),
                                "profile": np.array([layer_profile(b) for b in calib])}}
            for name in eval_sets():
                cl = layer_clouds(weights, load_dataset(name, "test", n=2000)[0])
                bd = batch_diagrams(cl, m)
                res[m][name] = {"sw": np.array([sw_to_anchors(b, anchors) for b in bd]),
                                "profile": np.array([layer_profile(b) for b in bd])}
            print(f"  computed diagrams and SW distances for m = {m}", flush=True)
        torch.save(res, cache)

    m5 = json.loads((RESULTS_DIR / f"m5_data_manifold_s{seed}.json").read_text())["batch"]
    results = {}
    for m in BATCH_SIZES:
        cal = res[m]["calib"]
        mu_sw, sd_sw = cal["sw"].mean(0), cal["sw"].std(0) + 1e-12
        mu_pr, sd_pr = cal["profile"].mean(0), cal["profile"].std(0) + 1e-12
        sc = {"SW to reference": {n: ((res[m][n]["sw"] - mu_sw) / sd_sw).mean(1) for n in eval_sets()},
              "layer profile": {n: (((res[m][n]["profile"] - mu_pr) / sd_pr) ** 2).mean(1) for n in eval_sets()}}
        rows = []
        print(f"\nbatch size m = {m}: {2000 // m} ID vs {2000 // m} OOD batches (AUROC / FPR95)")
        print(f"{'OOD set':14s} | {'SW to reference':>15s} | {'layer profile':>15s} | {'[5] descriptors':>15s} | {'TU batch mean':>15s}")
        m5rows = {r["ood"]: r for r in m5[str(m)]["rows"]}
        for n in eval_sets()[1:]:
            r = {"ood": n}
            for k in sc:
                a, f = ood_metrics(sc[k][ID], sc[k][n])
                r[k] = {"auroc": a, "fpr95": f}
            rows.append(r)
            print(f"{n:14s} | " + " | ".join(f"{r[k]['auroc']:6.1f} / {r[k]['fpr95']:5.1f}" for k in sc)
                  + f" | {m5rows[n]['manifold']['auroc']:6.1f} / {m5rows[n]['manifold']['fpr95']:5.1f}"
                  + f" | {m5rows[n]['tu']['auroc']:6.1f} / {m5rows[n]['tu']['fpr95']:5.1f}")
        avg = {k: {q: float(np.mean([r[k][q] for r in rows])) for q in ("auroc", "fpr95")} for k in sc}
        a5, atu = m5[str(m)]["average"]["manifold"], m5[str(m)]["average"]["tu"]
        print(f"{'average':14s} | " + " | ".join(f"{avg[k]['auroc']:6.1f} / {avg[k]['fpr95']:5.1f}" for k in sc)
              + f" | {a5['auroc']:6.1f} / {a5['fpr95']:5.1f} | {atu['auroc']:6.1f} / {atu['fpr95']:5.1f}")
        results[str(m)] = {"rows": rows, "average": avg}

    # mean SW distance to the reference, per layer and dimension (m = 50)
    feat = [f"{l}:H{d}" for l in LAYER_NAMES for d in (0, 1)]
    print(f"\nmean SW distance to {ID}-train reference diagrams, m = 50:")
    print(f"  {'set':16s}" + "".join(f"{f:>14s}" for f in feat))
    sw_means = {}
    for n in ["calibration (train)"] + eval_sets():
        v = (res[50]["calib"] if n.startswith("calibration") else res[50][n])["sw"].mean(0)
        sw_means[n] = v.tolist()
        print(f"  {n:16s}" + "".join(f"{x:14.2f}" for x in v))
    pairs = [f"{a}->{b}:H{d}" for a, b in zip(LAYER_NAMES[:-1], LAYER_NAMES[1:]) for d in (0, 1)]
    print("\nmean layer-to-layer SW profile, m = 50:")
    print(f"  {'set':16s}" + "".join(f"{p:>20s}" for p in pairs))
    prof_means = {}
    for n in ["calibration (train)"] + eval_sets():
        v = (res[50]["calib"] if n.startswith("calibration") else res[50][n])["profile"].mean(0)
        prof_means[n] = v.tolist()
        print(f"  {n:16s}" + "".join(f"{x:20.2f}" for x in v))

    (RESULTS_DIR / f"m6_activation_ph_s{seed}.json").write_text(json.dumps(
        {"seed": seed, "anchors": N_ANCHORS, "directions": N_DIRECTIONS, "results": results,
         "sw_means_m50": sw_means, "profile_means_m50": prof_means,
         "sw_features": feat, "profile_features": pairs}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, d in zip(axes, (0, 1)):
        for n in ["calibration (train)"] + eval_sets():
            v = np.array(sw_means[n])[d::2]
            ax.plot(LAYER_NAMES, v, marker="o", lw=2.2 if n in (ID, "calibration (train)") else 1,
                    ls="--" if n.startswith("calibration") else "-",
                    label=f"{ID} train" if n.startswith("calibration") else n)
        ax.set_yscale("log")
        ax.set_ylabel(f"mean SW distance to {ID}-train PDs")
        ax.set_title(f"[6] H{d} diagrams: SW distance to reference per layer (m = 50)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"m6_activation_ph_s{seed}.png", dpi=150)


if __name__ == "__main__":
    main()
