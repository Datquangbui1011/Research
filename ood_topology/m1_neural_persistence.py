"""Method [1]: Neural Persistence (Rieck et al., ICLR 2019) used for OOD detection.

(a) Original NP: computed from the trained weights only (no input). It is one number per
    network, so every input gets the same score and it cannot rank samples (AUROC = 50).
(b) Input-dependent NP (our adaptation): the same NP formula applied to the activation graph
    |W_l(j,i) x_l(i)| of each input (the graph of [2]). OOD score = |NP(x) - mu_k| / sigma_k,
    with mu_k, sigma_k the NP mean/std of training inputs predicted as class k.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (N_TRAIN_REF, RESULTS_DIR, bipartite_mst_weights, eval_sets, load_dataset,
                    load_model, mean_net_pds, msp_score, ood_metrics)


def normalized_np(pds, min_edges, p=2):
    """Mean normalized NP over layers.

    pds: list over layers of (B, N_l) MST weights (decreasing); min_edges: list of (B,) minimum
    edge weight per layer. Weights are divided by the largest weight in the whole network
    (Rieck Alg. 1); PD points are (1, w'), persistence 1 - w'; NP_l = ||1 - w'||_p is divided by
    its upper bound (max w' - min w') * (|V_l| - 1)^(1/p) (Rieck Thm. 1) and averaged over layers.
    """
    gmax = torch.stack([pd[:, 0] for pd in pds], 1).max(1).values.clamp_min(1e-12)
    per_layer = []
    for pd, mn in zip(pds, min_edges):
        w = pd / gmax[:, None]
        np_l = ((1 - w) ** p).sum(1) ** (1 / p)
        bound = (w[:, 0] - mn / gmax) * (pd.shape[1]) ** (1 / p)  # |V_l| - 1 = #MST edges
        per_layer.append(np_l / bound.clamp_min(1e-12))
    per_layer = torch.stack(per_layer, 1)
    return per_layer.mean(1), per_layer


def min_edge_weights(weights, X):
    """Smallest activation-graph edge |W(j,i) x(i)| of every layer (needed for the NP bound)."""
    from common import forward_trace
    inputs, _ = forward_trace(weights, X)
    return [(h.abs() * w.abs().min(0).values[None]).min(1).values for h, (w, _) in zip(inputs, weights)]


def input_np(seed, name, split, n):
    model = load_model(seed)
    X, _ = load_dataset(name, split, n=n)
    d = mean_net_pds(seed, name, split, n)
    with torch.no_grad():
        mins = min_edge_weights(model.sample_weights(mean=True), X)
    npv, per_layer = normalized_np(d["pds"], mins)
    return npv.numpy(), per_layer.numpy(), d["logits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed = args.seed

    # (a) original, weight-only NP
    model = load_model(seed)
    with torch.no_grad():
        W = model.sample_weights(mean=True)
        pds = [bipartite_mst_weights(w.abs().t()[None]) for w, _ in W]
        mins = [w.abs().min()[None] for w, _ in W]
    np_orig, np_orig_layers = normalized_np(pds, mins)
    print(f"[1a] original NP (weights only): network NP = {np_orig.item():.4f}, "
          f"per layer = {np.round(np_orig_layers.numpy()[0], 4).tolist()}")
    print("     -> identical for every input, so as an OOD score AUROC = 50.0 on every set\n")

    # (b) input-dependent NP on activation graphs
    tr_np, _, tr_logits = input_np(seed, "MNIST", "train", N_TRAIN_REF)
    tr_pred = tr_logits.argmax(1).numpy()
    mu = np.array([tr_np[tr_pred == k].mean() for k in range(10)])
    sd = np.array([tr_np[tr_pred == k].std() for k in range(10)])

    def score(npv, logits):
        k = logits.argmax(1).numpy()
        return np.abs(npv - mu[k]) / sd[k]

    vals, scores, msp = {}, {}, {}
    for name in eval_sets():
        npv, _, logits = input_np(seed, name, "test", 2000)
        vals[name], scores[name], msp[name] = npv, score(npv, logits), msp_score(logits)

    rows = []
    print(f"{'OOD set':14s} {'NP mean':>8s} | {'NP AUROC':>8s} {'FPR95':>6s} | {'MSP AUROC':>9s} {'FPR95':>6s}")
    print(f"{'MNIST (ID)':14s} {vals['MNIST'].mean():8.4f} |")
    for name in eval_sets()[1:]:
        a, f = ood_metrics(scores["MNIST"], scores[name])
        ab, fb = ood_metrics(msp["MNIST"], msp[name])
        rows.append({"ood": name, "np_mean": float(vals[name].mean()), "auroc": a, "fpr95": f,
                     "msp_auroc": ab, "msp_fpr95": fb})
        print(f"{name:14s} {vals[name].mean():8.4f} | {a:8.1f} {f:6.1f} | {ab:9.1f} {fb:6.1f}")
    avg = {k: float(np.mean([r[k] for r in rows])) for k in ("auroc", "fpr95", "msp_auroc", "msp_fpr95")}
    print(f"{'average':14s} {'':8s} | {avg['auroc']:8.1f} {avg['fpr95']:6.1f} | {avg['msp_auroc']:9.1f} {avg['msp_fpr95']:6.1f}")

    out = {"seed": seed, "np_original": float(np_orig.item()),
           "np_original_per_layer": np_orig_layers.numpy()[0].tolist(),
           "np_train_mean": float(tr_np.mean()), "rows": rows, "average": avg}
    (RESULTS_DIR / f"m1_neural_persistence_s{seed}.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(v.min() for v in vals.values()), max(v.max() for v in vals.values()), 60)
    for name, v in vals.items():
        ax.hist(v, bins=bins, histtype="step", density=True, lw=2 if name == "MNIST" else 1,
                label=name + (" (ID)" if name == "MNIST" else ""))
    ax.axvline(np_orig.item(), color="k", ls="--", lw=1, label="original NP (weights only)")
    ax.set_xlabel("mean normalized neural persistence of the activation graphs")
    ax.set_ylabel("density")
    ax.set_title("[1] Neural persistence per input (MLP 784-64-16-10 on MNIST)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"m1_neural_persistence_s{seed}.png", dpi=150)


if __name__ == "__main__":
    main()
