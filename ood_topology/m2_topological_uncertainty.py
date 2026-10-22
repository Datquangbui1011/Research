"""Method [2]: Topological Uncertainty (Lacombe et al., IJCAI 2021) for OOD detection.

For an input x with predicted class k:
  - D_l(x): 0-dim PD of the activation graph |W_l(j,i) x_l(i)| of layer l under the superlevel
    filtration = the N_l = d_l + d_{l+1} - 1 MST weights (a 1-D measure with fixed size).
  - D_{l,k}: Frechet mean of the training PDs predicted as class k. For 1-D diagrams with the
    same number of points this is the coordinate-wise mean of the sorted weights.
  - TU(x) = (1/L) sum_l Dist(D_l(x), D_{l,k}), Dist = 2-Wasserstein distance between the two
    measures = L2 distance between the sorted weight vectors.
A sample is flagged OOD when TU is large (paper: above a quantile of the training TUs).
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import N_TRAIN_REF, RESULTS_DIR, eval_sets, mean_net_pds, msp_score, ood_metrics


def class_mean_pds(pds, pred, n_classes=10):
    """Frechet mean PD per layer and predicted class: list over layers of (K, N_l)."""
    return [torch.stack([pd[pred == k].mean(0) for k in range(n_classes)]) for pd in pds]


def tu_score(pds, logits, means):
    """Returns (TU averaged over layers, per-layer distances (B, L))."""
    k = logits.argmax(1)
    per_layer = torch.stack([(pd - m[k]).norm(dim=1) for pd, m in zip(pds, means)], 1)
    return per_layer.mean(1).numpy(), per_layer.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed

    tr = mean_net_pds(seed, "MNIST", "train", N_TRAIN_REF)
    means = class_mean_pds(tr["pds"], tr["logits"].argmax(1))
    tu_train, tr_layers = tu_score(tr["pds"], tr["logits"], means)
    layer_scale = tr_layers.mean(0)  # diagnostic: put every layer's distance on the same scale

    tu, layers, msp, conf = {}, {}, {}, {}
    for name in eval_sets():
        d = mean_net_pds(seed, name, "test", 2000)
        tu[name], layers[name] = tu_score(d["pds"], d["logits"], means)
        msp[name] = msp_score(d["logits"])
        conf[name] = 1 - msp[name]

    L = layers["MNIST"].shape[1]
    rows = []
    hdr = " ".join(f"{'L' + str(l + 1):>5s}" for l in range(L))
    print(f"{'OOD set':14s} {'TU mean':>8s} {'conf':>5s} | {'TU AUROC':>8s} {'FPR95':>6s} | "
          f"{'TUnorm':>6s} {'FPR95':>6s} | per-layer AUROC {hdr} | {'MSP AUROC':>9s} {'FPR95':>6s}")
    print(f"{'MNIST (ID)':14s} {tu['MNIST'].mean():8.4f} {conf['MNIST'].mean():5.3f} |  (train TU mean {tu_train.mean():.4f})")
    for name in eval_sets()[1:]:
        a, f = ood_metrics(tu["MNIST"], tu[name])
        an, fn = ood_metrics((layers["MNIST"] / layer_scale).mean(1), (layers[name] / layer_scale).mean(1))
        al = [ood_metrics(layers["MNIST"][:, l], layers[name][:, l])[0] for l in range(L)]
        ab, fb = ood_metrics(msp["MNIST"], msp[name])
        rows.append({"ood": name, "tu_mean": float(tu[name].mean()), "confidence": float(conf[name].mean()),
                     "auroc": a, "fpr95": f, "norm_auroc": an, "norm_fpr95": fn, "layer_auroc": al, "msp_auroc": ab, "msp_fpr95": fb})
        print(f"{name:14s} {tu[name].mean():8.4f} {conf[name].mean():5.3f} | {a:8.1f} {f:6.1f} | {an:6.1f} {fn:6.1f} | "
              f"{'':15s} " + " ".join(f"{x:5.1f}" for x in al) + f" | {ab:9.1f} {fb:6.1f}")
    keys = ("auroc", "fpr95", "norm_auroc", "norm_fpr95", "msp_auroc", "msp_fpr95")
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    avg["layer_auroc"] = np.mean([r["layer_auroc"] for r in rows], 0).tolist()
    print(f"{'average':14s} {'':14s} | {avg['auroc']:8.1f} {avg['fpr95']:6.1f} | {avg['norm_auroc']:6.1f} {avg['norm_fpr95']:6.1f} | {'':15s} "
          + " ".join(f"{x:5.1f}" for x in avg["layer_auroc"]) + f" | {avg['msp_auroc']:9.1f} {avg['msp_fpr95']:6.1f}")
    (RESULTS_DIR / f"m2_topological_uncertainty_s{seed}.json").write_text(
        json.dumps({"seed": seed, "tu_train_mean": float(tu_train.mean()),
                    "tu_id_mean": float(tu["MNIST"].mean()), "rows": rows, "average": avg}, indent=2))

    # Figure: TU distributions (like Fig. 4 of the paper) and TU vs. confidence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    hi = np.quantile(np.concatenate(list(tu.values())), 0.995)
    bins = np.linspace(0, hi, 60)
    for name, v in tu.items():
        axes[0].hist(v, bins=bins, histtype="step", density=True, lw=2.2 if name == "MNIST" else 1,
                     label=name + (" (ID)" if name == "MNIST" else ""))
    axes[0].set_xlabel("Topological Uncertainty")
    axes[0].set_ylabel("density")
    axes[0].set_title("[2] TU distribution per dataset")
    axes[0].legend(fontsize=7)
    for name in eval_sets():
        axes[1].scatter(conf[name][:300], tu[name][:300], s=4, alpha=0.5,
                        label=name + (" (ID)" if name == "MNIST" else ""))
    axes[1].set_xlabel("softmax confidence (max probability)")
    axes[1].set_ylabel("Topological Uncertainty")
    axes[1].set_ylim(0, hi)
    axes[1].set_title("TU vs. confidence (300 samples per set)")
    axes[1].legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"m2_topological_uncertainty_s{seed}.png", dpi=150)


if __name__ == "__main__":
    main()
