"""Aggregate Phase 1 (several in-distribution sets) into mean ± std tables and a heatmap.

Usage: python aggregate_phase1.py [--ids MNIST FashionMNIST KMNIST SVHN] [--seeds 0 1 2]
Writes results/phase1/summary_phase1.md and results/phase1/summary_phase1.png.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate import PER_IMAGE, batch_extractors
from common import DATA_CACHE, OOD_GROUPS, ROOT

P1 = ROOT / "results" / "phase1"
GROUPS = ("near", "far", "shift")
GROUP_TITLE = {"near": "near-OOD", "far": "far-OOD", "shift": "cov. shift"}
# per-image scores shown in the main tables (label in aggregate.PER_IMAGE)
MAIN = ["MSP (max softmax)", "Bayesian mutual information", "k-NN distance †", "[1] NP, per input (adaptation)",
        "[2] TU (paper)", "[2] TU, layers rescaled", "[3] pTU (paper)", "[4] PL full spectrum †",
        "[4] β₀ only (PH part) †", "[5] local ΔH0 only †"]


def collect(spec, ids, seeds):
    """-> {label: {id: {ood: array (seeds, 2)}}} for every extractor in spec."""
    out = {}
    for label, f, ext in spec:
        out[label] = {}
        for i in ids:
            per_seed = [ext(json.loads((P1 / i / f"{f}_s{s}.json").read_text())) for s in seeds]
            out[label][i] = {o: np.array([d[o] for d in per_seed]) for o in per_seed[0]}
    return out


def group_mean(d, i, group, k=0):
    """Per-seed average of metric k over the OOD sets of a group -> array (seeds,)."""
    names = [n for n in OOD_GROUPS[i][group] if n in d]
    return np.mean([d[n][:, k] for n in names], 0)


def pm(x, d=1):
    return f"{np.mean(x):.{d}f} ± {np.std(x):.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=list(OOD_GROUPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    ids, seeds = args.ids, args.seeds

    pi = collect([p for p in PER_IMAGE if p[0] in MAIN], ids, seeds)
    lines = [f"# Phase 1: several in-distribution datasets", "",
             f"Bayesian MLP 784-64-16-10 trained on each ID set; mean ± std over {len(seeds)} seeds. "
             "Each cell first averages AUROC over the OOD sets of the group (near-OOD / far-OOD / covariate "
             "shift), then over seeds. † = first 1000 images per set.", ""]

    lines += ["## Test accuracy", "", "| ID set | posterior mean | 30-sample average |", "|---|---|---|"]
    for i in ids:
        acc = np.array([[float(v) for v in (DATA_CACHE / "phase1" / i / f"bnn_seed{s}_acc.txt").read_text().split()]
                        for s in seeds])
        lines.append(f"| {i} | {pm(100 * acc[:, 0], 2)} % | {pm(100 * acc[:, 1], 2)} % |")

    lines += ["", "## OOD sets per ID set", "", "| ID set | near-OOD | far-OOD | covariate shift |", "|---|---|---|---|"]
    for i in ids:
        lines.append(f"| {i} | " + " | ".join(", ".join(OOD_GROUPS[i][g]) for g in GROUPS) + " |")

    # one table per ID set
    for i in ids:
        lines += ["", f"## ID = {i}: per-image AUROC ↑ (FPR95 ↓)", "",
                  "| Method | near-OOD | far-OOD | cov. shift |", "|---|---|---|---|"]
        for label in MAIN:
            d = pi[label][i]
            lines.append(f"| {label} | " + " | ".join(
                f"{pm(group_mean(d, i, g, 0))} ({pm(group_mean(d, i, g, 1))})" for g in GROUPS) + " |")
        lines += ["", f"Per OOD set (AUROC):", "",
                  "| Method | " + " | ".join(n for g in GROUPS for n in OOD_GROUPS[i][g]) + " |",
                  "|---|" + "---|" * sum(len(OOD_GROUPS[i][g]) for g in GROUPS)]
        for label in MAIN:
            d = pi[label][i]
            lines.append(f"| {label} | " + " | ".join(
                pm(d[n][:, 0]) for g in GROUPS for n in OOD_GROUPS[i][g]) + " |")

    # cross-dataset summary (per-image)
    lines += ["", "## Summary across ID sets: per-image AUROC (near / far / shift)", "",
              "| Method | " + " | ".join(ids) + " | mean over ID sets |", "|---|" + "---|" * (len(ids) + 1)]
    heat = np.zeros((len(MAIN), len(ids) * len(GROUPS)))
    for r, label in enumerate(MAIN):
        cells, allv = [], []
        for c, i in enumerate(ids):
            v = [np.mean(group_mean(pi[label][i], i, g)) for g in GROUPS]
            heat[r, c * len(GROUPS):(c + 1) * len(GROUPS)] = v
            allv.append(v)
            cells.append(" / ".join(f"{x:.1f}" for x in v))
        m = np.mean(allv, 0)
        lines.append(f"| {label} | " + " | ".join(cells) + " | " + " / ".join(f"**{x:.1f}**" for x in m) + " |")

    # batch level
    for m in (10, 20):
        bt = collect(batch_extractors(m), ids, seeds)
        lines += ["", f"## Batch level, m = {m} images: AUROC (near / far / shift)", "",
                  "| Method | " + " | ".join(ids) + " |", "|---|" + "---|" * len(ids)]
        for label in bt:
            lines.append(f"| {label} | " + " | ".join(
                " / ".join(f"{np.mean(group_mean(bt[label][i], i, g)):.1f}" for g in GROUPS) for i in ids) + " |")

    # pTU permutation test
    lines += ["", "## [3] pTU permutation test: % of repeats rejecting H0 (α = 0.05), m = 20 / 50", "",
              "| ID set | ID test (false alarm) | near-OOD | far-OOD | cov. shift |", "|---|---|---|---|---|"]
    for i in ids:
        t = [json.loads((P1 / i / f"m3_ptu_s{s}.json").read_text())["permutation_test"] for s in seeds]

        def rate(names, mm):
            return np.mean([[tt[n][str(mm)]["reject_rate"] for n in names] for tt in t])
        cells = [f"{rate([i], 20):.0f} / {rate([i], 50):.0f}"]
        cells += [f"{rate(OOD_GROUPS[i][g], 20):.0f} / {rate(OOD_GROUPS[i][g], 50):.0f}" for g in GROUPS]
        lines.append(f"| {i} | " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    (P1 / "summary_phase1.md").write_text(md)
    print(md)

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * heat.shape[1], 0.5 + 0.42 * len(MAIN)))
    im = ax.imshow(heat, cmap="RdYlGn", vmin=40, vmax=100, aspect="auto")
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            ax.text(c, r, f"{heat[r, c]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(MAIN)), MAIN, fontsize=8)
    ax.set_xticks(range(heat.shape[1]), [GROUP_TITLE[g] for _ in ids for g in GROUPS], fontsize=7, rotation=45, ha="right")
    for c, i in enumerate(ids):
        ax.text(c * len(GROUPS) + 1, -0.9, f"ID = {i}", ha="center", fontsize=9, fontweight="bold")
        if c:
            ax.axvline(c * len(GROUPS) - 0.5, color="k", lw=1.5)
    fig.colorbar(im, ax=ax, label="AUROC (mean over 3 seeds)", shrink=0.8)
    ax.set_title("Phase 1: per-image OOD detection across in-distribution sets", pad=22)
    fig.tight_layout()
    fig.savefig(P1 / "summary_phase1.png", dpi=150)


if __name__ == "__main__":
    main()
