"""Method [3]: predictive Topological Uncertainty, pTU (Yeh & Yang, 2025) for OOD detection.

Draw S = 30 weight samples theta_1..theta_S from the Bayes-by-Backprop posterior (the same
samples for every input). For input x and layer l, D_l^s(x) is the 0-dim PD (sorted MST weights)
of the activation graph of F_{theta_s}. pTU(x) = (1/L) sum_l FrechetVar_s(D_l^s(x)) (Def. 3.1).
With the 2-Wasserstein distance on 1-D diagrams of equal size, the Frechet mean is the
coordinate-wise mean of the sorted vectors, so FrechetVar = mean_s ||D_l^s - mean_s D_l^s||^2.

Evaluated two ways:
  (a) per sample: pTU as an OOD score (AUROC / FPR95), against Bayesian baselines computed
      from the same 30 samples (predictive entropy, mutual information) and MSP;
  (b) per dataset, as in the paper: Algorithm 2, a permutation test with the empirical
      Wasserstein distance between the pTU values of m training inputs and m test inputs.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CACHE_DIR, RESULTS_DIR, activation_graphs, bipartite_mst_weights, eval_sets,
                    load_dataset, load_model, ood_metrics)

S = 30
N_TRAIN_PTU = 2000  # training inputs whose pTU forms the reference distribution


@torch.no_grad()
def ptu_and_bayes(model, X, posterior, dev, chunk=1000):
    """pTU (B,), per-layer variances (B, L), MC-averaged probabilities (B, K), mutual information (B,)."""
    out_ptu, out_layer, out_prob, out_mi = [], [], [], []
    for c in range(0, len(X), chunk):
        x = X[c:c + chunk].to(dev)
        pds_all = None  # per layer: (S, B, N_l), two-pass Frechet variance (float32 on MPS)
        probs, ent_each = 0, 0
        for s, weights in enumerate(posterior):
            graphs, _, logits = activation_graphs(weights, x)
            pds = [bipartite_mst_weights(G) for G in graphs]
            if pds_all is None:
                pds_all = [torch.empty(len(posterior), *p.shape, device=dev) for p in pds]
            for l, p in enumerate(pds):
                pds_all[l][s] = p
            pr = torch.softmax(logits, 1)
            probs = probs + pr / len(posterior)
            ent_each = ent_each - (pr * pr.clamp_min(1e-12).log()).sum(1) / len(posterior)
        var = torch.stack([((P - P.mean(0)) ** 2).sum(2).mean(0) for P in pds_all], 1)
        del pds_all
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(1)
        out_ptu.append(var.mean(1).cpu())
        out_layer.append(var.cpu())
        out_prob.append(probs.cpu())
        out_mi.append((ent - ent_each).cpu())
    return [torch.cat(o).float().numpy() for o in (out_ptu, out_layer, out_prob, out_mi)]


def wasserstein_1d(a, b, p=1):
    """Discrete p-Wasserstein between equal-size samples via order statistics (paper, Sec. 4)."""
    return (np.abs(np.sort(a) - np.sort(b)) ** p).sum() ** (1 / p)


def permutation_test(x, y, n_perm=1000, rng=None):
    """Algorithm 2 of the pTU paper."""
    rng = rng or np.random.default_rng(0)
    t_obs = wasserstein_1d(x, y)
    z = np.concatenate([x, y])
    t = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(z)
        t[i] = wasserstein_1d(z[:len(x)], z[len(x):])
    return float(np.mean(t >= t_obs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    seed = ap.parse_args().seed
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    cache = CACHE_DIR / f"pd_ptu_s{seed}.pt"
    if cache.exists():
        res = torch.load(cache, weights_only=False)
    else:
        model = load_model(seed, dev)
        torch.manual_seed(1000 + seed)
        posterior = [model.sample_weights() for _ in range(S)]
        res = {}
        # reference pTU of training inputs: a training subset disjoint from the 10k used in [1]/[2]
        Xtr, _ = load_dataset("MNIST", "train")
        Xtr = Xtr[torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(7))[-N_TRAIN_PTU:]]
        for name, X in [("train", Xtr)] + [(n, load_dataset(n, "test", n=2000)[0]) for n in eval_sets()]:
            res[name] = dict(zip(("ptu", "layers", "probs", "mi"), ptu_and_bayes(model, X, posterior, dev)))
            print(f"  computed pTU for {name}", flush=True)
        torch.save(res, cache)

    def entropy(p):
        return -(p * np.log(np.clip(p, 1e-12, None))).sum(1)

    scores = {n: {"pTU": r["ptu"], "Entropy": entropy(r["probs"]), "MI": r["mi"],
                  "MSP (BMA)": 1 - r["probs"].max(1)} for n, r in res.items()}
    L = res["MNIST"]["layers"].shape[1]

    # (a) per-sample OOD detection
    methods = ["pTU", "MI", "Entropy", "MSP (BMA)"]
    rows = []
    print(f"\n(a) per-sample  {'pTU mean':>9s} | " + " | ".join(f"{m:>9s} AUROC FPR95" for m in methods)
          + " | pTU per-layer AUROC")
    print(f"{'MNIST (ID)':14s} {res['MNIST']['ptu'].mean():9.5f} | (train pTU mean {res['train']['ptu'].mean():.5f})")
    for name in eval_sets()[1:]:
        r = {"ood": name, "ptu_mean": float(res[name]["ptu"].mean())}
        line = f"{name:14s} {r['ptu_mean']:9.5f} | "
        for m in methods:
            a, f = ood_metrics(scores["MNIST"][m], scores[name][m])
            r[m] = {"auroc": a, "fpr95": f}
            line += f"{'':9s} {a:5.1f} {f:5.1f} | "
        r["layer_auroc"] = [ood_metrics(res["MNIST"]["layers"][:, l], res[name]["layers"][:, l])[0] for l in range(L)]
        print(line + " ".join(f"{x:5.1f}" for x in r["layer_auroc"]))
        rows.append(r)
    avg = {m: {k: float(np.mean([r[m][k] for r in rows])) for k in ("auroc", "fpr95")} for m in methods}
    avg["layer_auroc"] = np.mean([r["layer_auroc"] for r in rows], 0).tolist()
    print(f"{'average':14s} {'':9s} | " + " | ".join(f"{'':9s} {avg[m]['auroc']:5.1f} {avg[m]['fpr95']:5.1f}" for m in methods)
          + " | " + " ".join(f"{x:5.1f}" for x in avg["layer_auroc"]))

    # (b) dataset-level permutation test (Algorithm 2): m training vs m test inputs, repeated
    rng = np.random.default_rng(seed)
    sizes, reps = [20, 50, 100], 50
    tests = {}
    print(f"\n(b) Algorithm 2 permutation test (W1 on pTU, 1000 permutations, alpha = 0.05, {reps} repeats)")
    print(f"{'test set':14s} " + " ".join(f"{'m=' + str(m):>16s}" for m in sizes) + "   (rejection rate % / median p)")
    for name in eval_sets():
        tests[name] = {}
        line = f"{name + (' (ID)' if name == 'MNIST' else ''):14s} "
        for m in sizes:
            pv = []
            for _ in range(reps):
                x = rng.choice(res["train"]["ptu"], m, replace=False)
                y = rng.choice(res[name]["ptu"], m, replace=False)
                pv.append(permutation_test(x, y, rng=rng))
            pv = np.array(pv)
            tests[name][m] = {"reject_rate": float(100 * np.mean(pv < 0.05)), "median_p": float(np.median(pv))}
            line += f"{tests[name][m]['reject_rate']:9.0f} / {tests[name][m]['median_p']:.3f}"
        print(line)

    (RESULTS_DIR / f"m3_ptu_s{seed}.json").write_text(json.dumps(
        {"seed": seed, "S": S, "ptu_train_mean": float(res["train"]["ptu"].mean()),
         "rows": rows, "average": avg, "permutation_test": tests}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    allv = np.concatenate([res[n]["ptu"] for n in eval_sets()])
    bins = np.logspace(np.log10(np.quantile(allv, 0.002)), np.log10(np.quantile(allv, 0.998)), 60)
    for name in eval_sets():
        axes[0].hist(res[name]["ptu"], bins=bins, histtype="step", density=True,
                     lw=2.2 if name == "MNIST" else 1, label=name + (" (ID)" if name == "MNIST" else ""))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("pTU (log scale)")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"[3] pTU distribution per dataset (S = {S} posterior samples)")
    axes[0].legend(fontsize=7)
    for name in eval_sets():
        axes[1].plot(sizes, [tests[name][m]["reject_rate"] for m in sizes], marker="o",
                     lw=2.2 if name == "MNIST" else 1, label=name + (" (ID)" if name == "MNIST" else ""))
    axes[1].axhline(5, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("samples per test set m")
    axes[1].set_ylabel("H0 rejected (%)")
    axes[1].set_title("Permutation test (Alg. 2): rejection rate at alpha = 0.05")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"m3_ptu_s{seed}.png", dpi=150)


if __name__ == "__main__":
    main()
