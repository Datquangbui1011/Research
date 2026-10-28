"""Run every OpenOOD checkpoint once over every image list and cache its features.
Full sets: logits + penultimate features. Subsamples (N_SUB per test set, 10 000 train images):
per-stage features for the point-cloud methods [5] and [6]."""
import time

import numpy as np

from bench import BENCHMARKS, SEEDS, features, images, ood_sets, subsample_idx

N_TRAIN_STAGES = 10000


def train_stage_idx(n):
    return np.sort(np.random.default_rng(99).choice(n, N_TRAIN_STAGES, replace=False))


def main():
    for bench in BENCHMARKS:
        for seed in SEEDS:
            t = time.time()
            for split in [f"train_{bench}", f"val_{bench}", f"test_{bench}"] + [f"test_{n}" for n in ood_sets(bench)]:
                features(bench, seed, split)
                n = len(images(bench, split)[0])
                idx = train_stage_idx(n) if split.startswith("train") else subsample_idx(n)
                if not split.startswith("val"):
                    features(bench, seed, split, stages=True, idx=idx)
            print(f"{bench} s{seed}: features cached in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
