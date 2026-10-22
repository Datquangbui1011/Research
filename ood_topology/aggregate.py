"""Aggregate the per-seed results of all six methods into mean ± std tables and a figure.

Usage: python aggregate.py --seeds 0 1 2   (writes results/summary_seeds.md and .png)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CACHE_DIR, OOD_SETS, RESULTS_DIR

# (label, file, extractor(json) -> {ood: (auroc, fpr95)}), † = first 1000 images per set
PER_IMAGE = [
    ("MSP (max softmax)", "m1_neural_persistence", lambda j: {r["ood"]: (r["msp_auroc"], r["msp_fpr95"]) for r in j["rows"]}),
    ("Bayesian mutual information", "m3_ptu", lambda j: {r["ood"]: (r["MI"]["auroc"], r["MI"]["fpr95"]) for r in j["rows"]}),
    ("k-NN distance †", "m5_data_manifold", lambda j: {r["ood"]: (r["kNN distance"]["auroc"], r["kNN distance"]["fpr95"]) for r in j["local_rows"]}),
    ("[1] NP, per input (adaptation)", "m1_neural_persistence", lambda j: {r["ood"]: (r["auroc"], r["fpr95"]) for r in j["rows"]}),
    ("[2] TU (paper)", "m2_topological_uncertainty", lambda j: {r["ood"]: (r["auroc"], r["fpr95"]) for r in j["rows"]}),
    ("[2] TU, layers rescaled", "m2_topological_uncertainty", lambda j: {r["ood"]: (r["norm_auroc"], r["norm_fpr95"]) for r in j["rows"]}),
    ("[3] pTU (paper)", "m3_ptu", lambda j: {r["ood"]: (r["pTU"]["auroc"], r["pTU"]["fpr95"]) for r in j["rows"]}),
    ("[4] PL full spectrum †", "m4_persistent_laplacian", lambda j: {r["ood"]: (r["PL full"]["auroc"], r["PL full"]["fpr95"]) for r in j["rows"]}),
    ("[4] β₀ only (PH part) †", "m4_persistent_laplacian", lambda j: {r["ood"]: (r["beta0 only (PH info)"]["auroc"], r["beta0 only (PH info)"]["fpr95"]) for r in j["rows"]}),
    ("[4] nonzero spectrum only †", "m4_persistent_laplacian", lambda j: {r["ood"]: (r["nonzero spectrum only"]["auroc"], r["nonzero spectrum only"]["fpr95"]) for r in j["rows"]}),
    ("[5] local ΔH0 + ΔH1 †", "m5_data_manifold", lambda j: {r["ood"]: (r["local PH (dH0, dH1)"]["auroc"], r["local PH (dH0, dH1)"]["fpr95"]) for r in j["local_rows"]}),
    ("[5] local ΔH0 only †", "m5_data_manifold", lambda j: {r["ood"]: (r["local dH0 only"]["auroc"], r["local dH0 only"]["fpr95"]) for r in j["local_rows"]}),
]


def batch_extractors(m):
    m = str(m)
    return [
        ("MSP, batch mean", "m5_data_manifold", lambda j: {r["ood"]: (r["msp"]["auroc"], r["msp"]["fpr95"]) for r in j["batch"][m]["rows"]}),
        ("[2] TU, batch mean", "m5_data_manifold", lambda j: {r["ood"]: (r["tu"]["auroc"], r["tu"]["fpr95"]) for r in j["batch"][m]["rows"]}),
        ("[5] H0/H1 persistence + intrinsic dim.", "m5_data_manifold", lambda j: {r["ood"]: (r["manifold"]["auroc"], r["manifold"]["fpr95"]) for r in j["batch"][m]["rows"]}),
        ("[6] SW distance to reference PDs", "m6_activation_ph", lambda j: {r["ood"]: (r["SW to reference"]["auroc"], r["SW to reference"]["fpr95"]) for r in j["results"][m]["rows"]}),
        ("[6] layer-to-layer SW profile", "m6_activation_ph", lambda j: {r["ood"]: (r["layer profile"]["auroc"], r["layer profile"]["fpr95"]) for r in j["results"][m]["rows"]}),
    ]


def collect(spec, seeds):
    """-> {label: array (seeds, n_ood, 2)}"""
    out = {}
    for label, f, ext in spec:
        per_seed = []
        for s in seeds:
            d = ext(json.loads((RESULTS_DIR / f"{f}_s{s}.json").read_text()))
            per_seed.append([d[o] for o in OOD_SETS])
        out[label] = np.array(per_seed)
    return out


def pm(x, d=1):
    return f"{np.mean(x):.{d}f} ± {np.std(x):.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    seeds = ap.parse_args().seeds
    n = len(seeds)
    lines = [f"Mean ± std over {n} training seeds ({', '.join(map(str, seeds))}). For each seed the metric is first "
             "averaged over the 7 OOD sets; the std is across seeds (population std). "
             "† = first 1000 images per set.", ""]

    acc = np.array([[float(v) for v in (CACHE_DIR / f"bnn_seed{s}_acc.txt").read_text().split()] for s in seeds])
    lines += [f"Test accuracy (posterior mean): {pm(100 * acc[:, 0], 2)} %; 30-sample average: {pm(100 * acc[:, 1], 2)} %.", ""]

    # per-image summary
    pi = collect(PER_IMAGE, seeds)
    lines += ["**Per-image OOD detection, average over the 7 OOD sets**", "",
              "| Approach | AUROC ↑ | FPR95 ↓ |", "|---|---|---|"]
    for label, v in pi.items():
        lines.append(f"| {label} | {pm(v[..., 0].mean(1))} | {pm(v[..., 1].mean(1))} |")

    # per-OOD-set AUROC for the main per-image scores
    main_rows = ["MSP (max softmax)", "[2] TU (paper)", "[3] pTU (paper)", "[4] β₀ only (PH part) †",
                 "[5] local ΔH0 only †", "k-NN distance †"]
    lines += ["", "**Per-image AUROC per OOD set**", "",
              "| Approach | " + " | ".join(OOD_SETS) + " |", "|---|" + "---|" * len(OOD_SETS)]
    for label in main_rows:
        v = pi[label][..., 0]
        lines.append(f"| {label} | " + " | ".join(pm(v[:, i]) for i in range(len(OOD_SETS))) + " |")

    # batch level
    batch = {}
    for m in (10, 20, 50):
        batch[m] = collect(batch_extractors(m), seeds)
    lines += ["", "**Batch-level OOD detection, average AUROC (FPR95) over the 7 OOD sets**", "",
              "| Approach | m = 10 | m = 20 | m = 50 |", "|---|---|---|---|"]
    for label in batch[10]:
        lines.append(f"| {label} | " + " | ".join(
            f"{pm(batch[m][label][..., 0].mean(1))} ({pm(batch[m][label][..., 1].mean(1))})" for m in (10, 20, 50)) + " |")

    # pTU permutation test
    lines += ["", "**[3] pTU permutation test (Algorithm 2): % of repeats rejecting H0 at α = 0.05**", "",
              "| Test set | m = 20 | m = 50 | m = 100 |", "|---|---|---|---|"]
    tests = [json.loads((RESULTS_DIR / f"m3_ptu_s{s}.json").read_text())["permutation_test"] for s in seeds]
    for name in ["MNIST"] + OOD_SETS:
        lines.append(f"| {name}{' (ID)' if name == 'MNIST' else ''} | " + " | ".join(
            pm([t[name][str(m)]["reject_rate"] for t in tests], 0) for m in (20, 50, 100)) + " |")

    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary_seeds.md").write_text(md)
    print(md)

    # figure: per-image and batch (m = 10) average AUROC with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    for ax, data, title in ((axes[0], pi, "Per-image OOD detection"),
                            (axes[1], batch[10], "Batch level, m = 10 images")):
        labels = list(data)
        means = [data[l][..., 0].mean(1).mean() for l in labels]
        stds = [data[l][..., 0].mean(1).std() for l in labels]
        colors = ["#9aa0a6" if not l.startswith("[") else "#3b6ea8" for l in labels]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, means, xerr=stds, color=colors, capsize=3)
        for yi, mu, sd in zip(y, means, stds):
            ax.text(mu + sd + 1, yi, f"{mu:.1f}", va="center", fontsize=8)
        ax.set_yticks(y, labels, fontsize=8)
        ax.axvline(50, color="k", lw=0.6, ls=":")
        ax.set_xlim(40, 105)
        ax.set_xlabel(f"average AUROC over 7 OOD sets (mean ± std, {n} seeds)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "summary_seeds.png", dpi=150)


if __name__ == "__main__":
    main()
