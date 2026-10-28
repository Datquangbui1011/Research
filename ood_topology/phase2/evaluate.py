"""Evaluate OOD scores on the OpenOOD v1.5 CIFAR benchmarks (3 checkpoints each).

A method is a function score(bench, seed) -> {split_name: scores} giving, for the ID test set and
every OOD test set, an array of scores where higher = more OOD. Methods marked `sub` are evaluated
on the fixed N_SUB subsample of every test set (cheaper; marked † in tables).
Results follow OpenOOD: metrics per OOD set, then averaged within near-OOD and far-OOD.
"""
import argparse
import json

import numpy as np
from scipy.special import logsumexp

from bench import (BENCHMARKS, RESULTS, SEEDS, features, fc_weights, group_of, ood_sets, openood_metrics,
                   subsample_idx)


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def test_splits(bench):
    return [f"test_{bench}"] + [f"test_{n}" for n in ood_sets(bench)]


# --------------------------------------------------------------------------- baselines
def _logits(bench, seed):
    return {s: features(bench, seed, s)["logits"] for s in test_splits(bench)}


def msp(bench, seed):
    out = {}
    for s, lg in _logits(bench, seed).items():
        p = np.exp(lg - logsumexp(lg, 1, keepdims=True))
        out[s] = -p.max(1)
    return out


def ebo(bench, seed):
    """Energy score (Liu et al. 2020), temperature 1."""
    return {s: -logsumexp(lg, 1) for s, lg in _logits(bench, seed).items()}


def mls(bench, seed):
    return {s: -lg.max(1) for s, lg in _logits(bench, seed).items()}


def _normed(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def knn(bench, seed, k=50):
    """KNN (Sun et al. 2022): distance to the k-th nearest normalized training feature (OpenOOD: K=50)."""
    import torch
    train = torch.from_numpy(_normed(features(bench, seed, f"train_{bench}")["feat"]))
    out = {}
    for s in test_splits(bench):
        q = torch.from_numpy(_normed(features(bench, seed, s)["feat"]))
        kth = []
        for c in range(0, len(q), 2000):
            d = torch.cdist(q[c:c + 2000], train)
            kth.append(d.topk(k, largest=False).values[:, -1])
        out[s] = torch.cat(kth).numpy()
    return out


def vim(bench, seed, dim=256):
    """ViM (Wang et al. 2022): energy minus alpha times the norm of the feature residual outside the
    principal subspace of training features (OpenOOD: dim = 256 for 512-d ResNet-18 features)."""
    W, b = fc_weights(bench, seed)
    u = -np.linalg.pinv(W) @ b
    tr = features(bench, seed, f"train_{bench}")
    ftr = tr["feat"].astype(np.float64) - u
    eigval, eigvec = np.linalg.eigh(np.cov(ftr.T))
    ns = eigvec[:, np.argsort(eigval)[::-1][dim:]]
    alpha = tr["logits"].max(1).mean() / np.linalg.norm(ftr @ ns, axis=1).mean()
    out = {}
    for s in test_splits(bench):
        d = features(bench, seed, s)
        r = np.linalg.norm((d["feat"].astype(np.float64) - u) @ ns, axis=1) * alpha
        out[s] = -(logsumexp(d["logits"], 1) - r)
    return out


METHODS = {"MSP": (msp, False), "EBO": (ebo, False), "MLS": (mls, False), "KNN": (knn, False), "VIM": (vim, False)}


# --------------------------------------------------------------------------- evaluation
def evaluate(name, fn, sub, bench, seed):
    """Subsample methods (sub=True) already return scores for the subsampled rows only."""
    scores = fn(bench, seed)
    sid = scores[f"test_{bench}"]
    rows = {n: openood_metrics(sid, scores[f"test_{n}"]) for n in ood_sets(bench)}
    groups = {g: {m: float(np.mean([rows[n][m] for n in rows if group_of(bench, n) == g]))
                  for m in ("auroc", "fpr95", "aupr_in", "aupr_out")} for g in ("near", "far")}
    return {"per_set": rows, "groups": groups}


def all_methods():
    from topo import TOPO_METHODS
    return {**METHODS, **TOPO_METHODS}


def run(names, benches=tuple(BENCHMARKS), seeds=SEEDS):
    methods = all_methods()
    for name in names:
        fn, sub = methods[name]
        for bench in benches:
            for seed in seeds:
                f = RESULTS / f"{slug(name)}_{bench}_s{seed}.json"
                res = evaluate(name, fn, sub, bench, seed)
                res.update({"method": name, "bench": bench, "seed": seed, "subsample": sub})
                f.write_text(json.dumps(res, indent=1))
            g = [json.loads((RESULTS / f"{slug(name)}_{bench}_s{s}.json").read_text())["groups"] for s in seeds]
            near = [x["near"]["auroc"] for x in g]
            far = [x["far"]["auroc"] for x in g]
            print(f"{name:18s} {bench:9s} near {np.mean(near):6.2f} (±{np.std(near):.2f})  "
                  f"far {np.mean(far):6.2f} (±{np.std(far):.2f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="*", default=None)
    ap.add_argument("--bench", nargs="+", default=list(BENCHMARKS))
    a = ap.parse_args()
    run(a.methods or list(all_methods()), a.bench)
