"""Generate the LaTeX tables and figures of the paper from the saved results.

Usage (from ood_topology/):  python paper/make_tables.py
Writes paper/tables/p1_*.tex, paper/tables/p2_main.tex and copies paper/figures/*.png.
Hand-written text (findings) lives in paper/tables/*_findings.tex and is not overwritten.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase2"))
from aggregate import PER_IMAGE, batch_extractors  # noqa: E402
from aggregate_phase1 import GROUPS, collect, group_mean  # noqa: E402
from common import OOD_GROUPS  # noqa: E402

TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
IDS = ["MNIST", "FashionMNIST", "KMNIST", "SVHN"]
SEEDS = [0, 1, 2]
SHORT = {"MNIST": "MNIST", "FashionMNIST": "F-MNIST", "KMNIST": "KMNIST", "SVHN": "SVHN"}
P1_ROWS = [  # (label in aggregate.PER_IMAGE, label in the paper)
    ("MSP (max softmax)", "MSP"),
    ("Bayesian mutual information", "Mutual information"),
    ("k-NN distance †", "$k$-NN distance$^\\dagger$"),
    ("[1] NP, per input (adaptation)", "[1] NP (per input)"),
    ("[2] TU (paper)", "[2] TU"),
    ("[2] TU, layers rescaled", "[2] TU-rescaled"),
    ("[3] pTU (paper)", "[3] pTU"),
    ("[4] PL full spectrum †", "[4] PL, full spectrum$^\\dagger$"),
    ("[4] β₀ only (PH part) †", "[4] PL, $\\beta_0$ only$^\\dagger$"),
    ("[5] local ΔH0 only †", "[5] local $\\Delta H_0$$^\\dagger$"),
]
NAME = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tin": "TIN", "mnist": "MNIST", "svhn": "SVHN",
        "texture": "Textures", "places365": "Places365"}


def bold_best(vals, fmt="{:.1f}", higher=True):
    v = np.array(vals, float)
    best = np.nanmax(v) if higher else np.nanmin(v)
    return [("\\textbf{" + fmt.format(x) + "}") if np.isclose(x, best) else fmt.format(x) for x in v]


def p1_sets():
    lines = ["\\begin{table}[t]", "\\centering\\small",
             "\\begin{tabular}{@{}l>{\\raggedright}p{3.4cm}>{\\raggedright}p{6.2cm}p{3.2cm}@{}}", "\\toprule",
             "ID set & near-OOD & far-OOD & covariate shift \\\\", "\\midrule"]
    for i in IDS:
        cells = [", ".join(n.replace("FashionMNIST", "F-MNIST").replace("TinyImageNet", "TIN") for n in OOD_GROUPS[i][g])
                 for g in GROUPS]
        lines.append(f"{SHORT[i]} & " + " & ".join(cells) + " \\tabularnewline")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Part I: OOD sets for each in-distribution set (2000 test images each).}",
              "\\label{tab:p1-sets}", "\\end{table}"]
    (TAB / "p1_sets.tex").write_text("\n".join(lines) + "\n")


def p1_main():
    labels = [p for p in PER_IMAGE if p[0] in dict(P1_ROWS)]
    pi = collect(labels, IDS, SEEDS)
    acc = []
    for i in IDS:
        a = [float((ROOT / "cache" / "phase1" / i / f"bnn_seed{s}_acc.txt").read_text().split()[0]) for s in SEEDS]
        acc.append(f"{100 * np.mean(a):.1f}")
    # matrix rows x (ID x group)
    M = np.array([[np.mean(group_mean(pi[lab][i], i, g)) for i in IDS for g in GROUPS] for lab, _ in P1_ROWS])
    avg = np.array([[M[r, [c * 3 + gi for c in range(len(IDS))]].mean() for gi in range(3)] for r in range(len(P1_ROWS))])
    full = np.hstack([M, avg])
    cols = [bold_best(full[:, c]) for c in range(full.shape[1])]
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{SHORT[i]} ({a}\\%)}}" for i, a in zip(IDS, acc))
    lines = ["\\begin{table*}[t]", "\\centering", "\\setlength{\\tabcolsep}{3pt}\\scriptsize",
             "\\begin{tabular}{@{}l" + "ccc" * (len(IDS) + 1) + "@{}}", "\\toprule",
             f" & {head} & \\multicolumn{{3}}{{c}}{{Mean over ID sets}} \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 3 * c}-{4 + 3 * c}}}" for c in range(len(IDS) + 1)),
             "Method & " + " & ".join(["Near", "Far", "Shift"] * (len(IDS) + 1)) + " \\\\", "\\midrule"]
    for r, (_, name) in enumerate(P1_ROWS):
        if r == 3:
            lines.append("\\midrule")
        lines.append(f"{{{name}}} & " + " & ".join(cols[c][r] for c in range(full.shape[1])) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Part I, per-image OOD detection: AUROC (\\%, mean over 3 seeds), averaged within each OOD "
              "group. ID test accuracy in parentheses. The first three rows are baselines; bold = best in the column. "
              "$^\\dagger$ = first 1000 images per set. Standard deviations and FPR@95 are in the repository.}",
              "\\label{tab:p1-main}", "\\end{table*}"]
    (TAB / "p1_main.tex").write_text("\n".join(lines) + "\n")


def p1_batch():
    rows = []
    for m in (10, 20):
        bt = collect(batch_extractors(m), IDS, SEEDS)
        for lab in bt:
            rows.append((m, lab, [np.mean(group_mean(bt[lab][i], i, g)) for i in IDS for g in GROUPS]))
    pretty = {"MSP, batch mean": "MSP (batch mean)", "[2] TU, batch mean": "[2] TU (batch mean)",
              "[5] H0/H1 persistence + intrinsic dim.": "[5] $P_0,P_1$, intrinsic dim.",
              "[6] SW distance to reference PDs": "[6] SW to reference",
              "[6] layer-to-layer SW profile": "[6] layer profile"}
    lines = ["\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{2.5pt}\\scriptsize",
             "\\begin{tabular}{@{}cl" + "ccc" * len(IDS) + "@{}}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{SHORT[i]}}}" for i in IDS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 3 * c}-{5 + 3 * c}}}" for c in range(len(IDS))),
             "$m$ & Method & " + " & ".join(["N", "F", "S"] * len(IDS)) + " \\\\", "\\midrule"]
    for m in (10, 20):
        block = [r for r in rows if r[0] == m]
        M = np.array([r[2] for r in block])
        cols = [bold_best(M[:, c], "{:.0f}") for c in range(M.shape[1])]
        for k, (_, lab, _) in enumerate(block):
            lead = f"\\multirow{{{len(block)}}}{{*}}{{{m}}}" if k == 0 else ""
            lines.append(f"{lead} & {{{pretty[lab]}}} & " + " & ".join(cols[c][k] for c in range(M.shape[1])) + " \\\\")
        if m == 10:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Part I, batch-level detection: AUROC (\\%) over batches of $m$ images from one source "
              "(N = near-OOD, F = far-OOD, S = covariate shift; mean over 3 seeds).}",
              "\\label{tab:p1-batch}", "\\end{table}"]
    (TAB / "p1_batch.tex").write_text("\n".join(lines) + "\n")


P2_OURS = [  # (results/phase2 file prefix, label, subsample?)
    ("MSP", "MSP (ours)", False), ("EBO", "EBO (ours)", False), ("MLS", "MLS (ours)", False),
    ("KNN", "KNN (ours)", False), ("VIM", "ViM (ours)", False),
    ("MI__Laplace", "Mutual information (Laplace)$^\\dagger$", True),
    ("1__NP", "[1] NP (per input)", False), ("2__TU", "[2] TU", False),
    ("3__pTU", "[3] pTU (Laplace)$^\\dagger$", True), ("3__pTU_two_sided", "[3] pTU two-sided$^\\dagger$", True),
    ("4__PL_full", "[4] PL, full spectrum$^\\dagger$", True), ("4__PL_beta0", "[4] PL, $\\beta_0$ only$^\\dagger$", True),
    ("5__local_dH0", "[5] local $\\Delta H_0$$^\\dagger$", True),
]
P2_PUBLISHED = ["MSP", "EBO", "MLS", "KNN", "VIM", "ReAct", "ASH", "RMDS", "ODIN", "MDS"]


def p2_main():
    res = ROOT / "results" / "phase2"
    pub = json.loads((ROOT / "phase2" / "openood_v15_table2.json").read_text())["methods"]
    lines = ["\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{4pt}\\small",
             "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Method & Near & Far & Near & Far \\\\", "\\midrule",
             "\\multicolumn{5}{@{}l}{\\emph{Published (OpenOOD v1.5, Table 2)}} \\\\"]
    for m in P2_PUBLISHED:
        v = pub[m]
        lines.append(f"{m} & " + " & ".join(f"{v[b][g][0]:.2f}" for b in ("cifar10", "cifar100") for g in ("near", "far")) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{5}{@{}l}{\\emph{This work (same checkpoints)}} \\\\"]
    for prefix, label, _ in P2_OURS:
        cells = []
        for b in ("cifar10", "cifar100"):
            files = [res / f"{prefix}_{b}_s{s}.json" for s in SEEDS]
            if not all(f.exists() for f in files):
                cells += ["--", "--"]
                continue
            g = [json.loads(f.read_text())["groups"] for f in files]
            for grp in ("near", "far"):
                a = [x[grp]["auroc"] for x in g]
                cells.append(f"{np.mean(a):.2f}\\,{{\\scriptsize$\\pm${np.std(a):.2f}}}")
        if label.startswith("Mutual") or label.startswith("[1]"):
            lines.append("\\midrule")
        lines.append(f"{{{label}}} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Part II, OpenOOD v1.5 CIFAR benchmarks with the released ResNet-18 checkpoints: "
              "per-image AUROC (\\%, mean $\\pm$ std over 3 checkpoints). The published rows are copied from "
              "OpenOOD v1.5; our re-implementations of the same baselines reproduce them. "
              "$^\\dagger$ = fixed subsample of 2000 images per test set (1000 for [4]).}",
              "\\label{tab:p2-main}", "\\end{table}"]
    (TAB / "p2_main.tex").write_text("\n".join(lines) + "\n")


def p2_batch():
    res = ROOT / "results" / "phase2"
    files = {b: [res / f"batch_{b}_s{s}.json" for s in SEEDS] for b in ("cifar10", "cifar100")}
    if not all(f.exists() for v in files.values() for f in v):
        (TAB / "p2_batch.tex").write_text("\\todo{Part II batch-level table: run phase2/batch.py}\n")
        return
    data = {b: [json.loads(f.read_text()) for f in v] for b, v in files.items()}
    names = list(data["cifar10"][0]["10"])
    lines = ["\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{4pt}\\small",
             "\\begin{tabular}{@{}clcccc@{}}", "\\toprule",
             " & & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{CIFAR-100} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "$m$ & Method & Near & Far & Near & Far \\\\", "\\midrule"]
    for m in ("10", "20"):
        M = np.array([[np.mean([d[m][n][g] for d in data[b]]) for b in ("cifar10", "cifar100") for g in ("near", "far")]
                      for n in names])
        cols = [bold_best(M[:, c]) for c in range(M.shape[1])]
        for k, n in enumerate(names):
            lead = f"\\multirow{{{len(names)}}}{{*}}{{{m}}}" if k == 0 else ""
            lines.append(f"{lead} & {{{n}}} & " + " & ".join(cols[c][k] for c in range(4)) + " \\\\")
        if m == "10":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Part II, batch-level detection: AUROC (\\%) over batches of $m$ images from one source "
              "(mean over 3 checkpoints).}", "\\label{tab:p2-batch}", "\\end{table}"]
    (TAB / "p2_batch.tex").write_text("\n".join(lines) + "\n")


def main():
    p1_sets()
    p1_main()
    p1_batch()
    p2_main()
    p2_batch()
    shutil.copy(ROOT / "results" / "phase1" / "summary_phase1.png", FIG / "summary_phase1.png")
    for f in ("p1_findings.tex", "p2_findings.tex"):
        if not (TAB / f).exists():
            (TAB / f).write_text("\\todo{findings}\n")
    print("tables written to", TAB)


if __name__ == "__main__":
    main()
