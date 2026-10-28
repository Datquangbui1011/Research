"""Batch-level OOD detection on the OpenOOD CIFAR benchmarks: is a batch of m images from one source OOD?

[5] descriptors (H0/H1 total persistence + TwoNN dimension of every stage's point cloud, spatial means)
[6] sliced-Wasserstein distance of every stage's Rips diagrams to training reference diagrams
    (Purvine et al.: one vector per image = channel vector at the position of largest activation norm)
    and the layer-to-layer SW profile; MSP and TU averaged over the batch for comparison.
Batches are cut from the fixed N_SUB subsample of every test set; references from 10 000 training images.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from m5_data_manifold import batches_of, zscore_score  # noqa: E402
from m6_activation_ph import batch_diagrams, layer_profile, sw_to_anchors  # noqa: E402

from bench import BENCHMARKS, RESULTS, SEEDS, features, group_of, ood_sets, openood_metrics, subsample_idx  # noqa: E402
from evaluate import msp, test_splits  # noqa: E402
from topo import topological_uncertainty  # noqa: E402

SIZES = (10, 20)
N_ANCHORS = 20


def clouds(d, kind):
    """Point clouds per layer: [5] uses spatial means + logits, [6] the max-norm channel vectors."""
    if kind == "mean":
        return [x.astype(np.float64) for x in d["stage_mean"]] + [d["logits"].astype(np.float64)]
    return [x.astype(np.float64) for x in d["stage_top"]]


def run(bench, seed):
    tr = features(bench, seed, f"train_{bench}", stages=True)
    test = {s: features(bench, seed, s, stages=True) for s in test_splits(bench)}
    full_msp, full_tu = msp(bench, seed), topological_uncertainty(bench, seed)
    out = {}
    for m in SIZES:
        # [5] reference descriptors from training batches
        ref5 = batches_of(clouds(tr, "mean"), m)
        # [6] anchors + calibration batches from training
        ref6 = batch_diagrams(clouds(tr, "top"), m)
        anchors, calib = ref6[:N_ANCHORS], ref6[N_ANCHORS:N_ANCHORS + 200]
        cal_sw = np.array([sw_to_anchors(b, anchors) for b in calib])
        cal_pr = np.array([layer_profile(b) for b in calib])
        sc = {"[5] descriptors": {}, "[6] SW to reference": {}, "[6] layer profile": {}, "MSP batch mean": {},
              "[2] TU batch mean": {}}
        for s, d in test.items():
            sc["[5] descriptors"][s] = zscore_score(ref5, batches_of(clouds(d, "mean"), m))
            bd = batch_diagrams(clouds(d, "top"), m)
            sc["[6] SW to reference"][s] = ((np.array([sw_to_anchors(b, anchors) for b in bd]) - cal_sw.mean(0))
                                            / (cal_sw.std(0) + 1e-12)).mean(1)
            sc["[6] layer profile"][s] = ((((np.array([layer_profile(b) for b in bd]) - cal_pr.mean(0))
                                             / (cal_pr.std(0) + 1e-12)) ** 2).mean(1))
            idx = subsample_idx(len(full_msp[s]))
            nb = len(idx) // m
            sc["MSP batch mean"][s] = full_msp[s][idx][:nb * m].reshape(nb, m).mean(1)
            sc["[2] TU batch mean"][s] = full_tu[s][idx][:nb * m].reshape(nb, m).mean(1)
        res = {}
        for name, v in sc.items():
            rows = {n: openood_metrics(v[f"test_{bench}"], v[f"test_{n}"]) for n in ood_sets(bench)}
            res[name] = {g: float(np.mean([rows[n]["auroc"] for n in rows if group_of(bench, n) == g]))
                         for g in ("near", "far")}
            res[name]["per_set"] = {n: rows[n]["auroc"] for n in rows}
        out[str(m)] = res
        print(f"{bench} s{seed} m={m}: " + "  ".join(f"{k} {v['near']:.1f}/{v['far']:.1f}" for k, v in res.items()), flush=True)
    (RESULTS / f"batch_{bench}_s{seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for bench in BENCHMARKS:
        for seed in SEEDS:
            run(bench, seed)
