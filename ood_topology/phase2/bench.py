"""Phase 2: the OpenOOD v1.5 CIFAR benchmarks with OpenOOD's released ResNet-18 checkpoints.

Benchmark definition, preprocessing and metrics follow OpenOOD v1.5 exactly
(openood/evaluation_api/datasets.py, preprocessor.py, evaluators/metrics.py):
  - ID: CIFAR-10 or CIFAR-100 (test split of the official image list)
  - near-OOD: CIFAR-100 / CIFAR-10 and TinyImageNet; far-OOD: MNIST, SVHN, Textures, Places365
  - Resize(32, bilinear) -> CenterCrop(32) -> ToTensor -> Normalize(dataset mean/std); grayscale -> RGB
  - 3 checkpoints per benchmark (s0, s1, s2)
Images are decoded once into uint8 caches; each checkpoint is run once and its features are cached:
logits, penultimate features (512) for every image, and per-stage features for a fixed subsample.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as skm

from resnet18_32x32 import ResNet18_32x32

ROOT = Path(__file__).resolve().parent.parent
OPENOOD = ROOT / "data" / "openood"
IMG_DIR = OPENOOD / "images_classic"
LIST_DIR = OPENOOD / "benchmark_imglist"
CKPT_DIR = OPENOOD / "checkpoints"
CACHE = ROOT / "cache" / "phase2"
RESULTS = ROOT / "results" / "phase2"
for d in (CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

BENCHMARKS = {
    "cifar10": {"classes": 10, "norm": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
                "near": ["cifar100", "tin"], "far": ["mnist", "svhn", "texture", "places365"]},
    "cifar100": {"classes": 100, "norm": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
                 "near": ["cifar10", "tin"], "far": ["mnist", "svhn", "texture", "places365"]},
}
SEEDS = [0, 1, 2]
N_SUB = 2000  # per-set subsample for the expensive methods (marked † in the tables)


def ood_sets(bench):
    return BENCHMARKS[bench]["near"] + BENCHMARKS[bench]["far"]


def group_of(bench, name):
    return "near" if name in BENCHMARKS[bench]["near"] else "far"


# --------------------------------------------------------------------------- images
def images(bench, split):
    """uint8 array (N, 32, 32, 3) and labels for an OpenOOD image list, e.g. split='test_tin'.
    Applies OpenOOD's Resize(32)+CenterCrop(32) before caching."""
    f = CACHE / f"img_{bench}_{split}.npz"
    if f.exists():
        d = np.load(f)
        return d["x"], d["y"]
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(32, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(32)])
    xs, ys = [], []
    for line in (LIST_DIR / bench / f"{split}.txt").read_text().splitlines():
        path, label = line.rsplit(" ", 1)
        img = tf(Image.open(IMG_DIR / path).convert("RGB"))
        xs.append(np.asarray(img, dtype=np.uint8))
        ys.append(int(label))
    x, y = np.stack(xs), np.array(ys)
    np.savez(f, x=x, y=y)
    return x, y


def subsample_idx(n, seed=0):
    """Fixed subsample (the same for every checkpoint) for the expensive methods."""
    rng = np.random.default_rng(1234 + seed)
    return np.sort(rng.choice(n, min(n, N_SUB), replace=False))


# --------------------------------------------------------------------------- model + features
def load_net(bench, seed, device):
    run = next(CKPT_DIR.glob(f"{bench}_resnet18_32x32_base_e100_lr0.1_default"))
    net = ResNet18_32x32(num_classes=BENCHMARKS[bench]["classes"])
    net.load_state_dict(torch.load(run / f"s{seed}" / "best.ckpt", map_location="cpu"))
    return net.to(device).eval()


def _stage_vectors(fmap):
    """Two summaries of a conv feature map (B, C, H, W): the spatial mean (used by [5], as in
    Magai & Ayzenberg) and the channel vector at the position with the largest norm (Purvine et al.)."""
    mean = fmap.mean((2, 3))
    flat = fmap.flatten(2)  # B, C, HW
    idx = flat.norm(dim=1).argmax(1)
    top = flat[torch.arange(len(flat)), :, idx]
    return mean, top


@torch.no_grad()
def features(bench, seed, split, stages=False, idx=None, device=None):
    """Cached features of one checkpoint on one image list.
    Returns dict: logits (N, K), feat (N, 512) [+ stage_mean / stage_top: lists of (N, C) for the
    4 residual stages + conv1, when stages=True (computed on the rows `idx`)]."""
    tag = f"{bench}_s{seed}_{split}" + ("_stages" if stages else "")
    f = CACHE / f"feat_{tag}.npz"
    if f.exists():
        d = np.load(f)
        out = {k: d[k] for k in d.files if not k.startswith("sm") and not k.startswith("st")}
        if stages:
            out["stage_mean"] = [d[f"sm{i}"] for i in range(5)]
            out["stage_top"] = [d[f"st{i}"] for i in range(5)]
        return out
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    net = load_net(bench, seed, device)
    x, y = images(bench, split)
    if idx is not None:
        x, y = x[idx], y[idx]
    mean, std = (torch.tensor(v, device=device).view(1, 3, 1, 1) for v in BENCHMARKS[bench]["norm"])
    logits, feats, sm, st = [], [], [[] for _ in range(5)], [[] for _ in range(5)]
    for s in range(0, len(x), 500):
        xb = torch.from_numpy(x[s:s + 500]).to(device).permute(0, 3, 1, 2).float().div(255)
        xb = (xb - mean) / std
        lg, flist = net(xb, return_feature_list=True)
        logits.append(lg.float().cpu())
        feats.append(flist[-1].flatten(1).float().cpu())
        if stages:
            for i, fm in enumerate(flist[:4]):
                a, b = _stage_vectors(fm)
                sm[i].append(a.cpu())
                st[i].append(b.cpu())
            sm[4].append(flist[-1].flatten(1).cpu())  # pooled layer4 = penultimate features
            st[4].append(lg.float().cpu())            # logits as the last "layer"
    out = {"logits": torch.cat(logits).numpy(), "feat": torch.cat(feats).numpy().astype(np.float16), "y": y}
    save = dict(out)
    if stages:
        out["stage_mean"] = [torch.cat(v).numpy() for v in sm]
        out["stage_top"] = [torch.cat(v).numpy() for v in st]
        save.update({f"sm{i}": v for i, v in enumerate(out["stage_mean"])})
        save.update({f"st{i}": v for i, v in enumerate(out["stage_top"])})
    np.savez(f, **save)
    return out


def fc_weights(bench, seed):
    net = load_net(bench, seed, "cpu")
    return net.fc.weight.detach().numpy(), net.fc.bias.detach().numpy()


# --------------------------------------------------------------------------- OpenOOD metrics
def openood_metrics(score_id, score_ood):
    """OpenOOD v1.5 metrics (evaluators/metrics.py). Input scores: higher = more OOD.
    OOD is the positive class; FPR@95 = fraction of ID samples flagged at the threshold that
    detects 95 % of OOD samples. Returns percentages: auroc, fpr95, aupr_in, aupr_out."""
    conf = -np.r_[np.asarray(score_id, float), np.asarray(score_ood, float)]  # OpenOOD: higher = ID
    ood = np.r_[np.zeros(len(score_id)), np.ones(len(score_ood))]
    fpr_l, tpr_l, _ = skm.roc_curve(ood, -conf)
    fpr95 = fpr_l[np.argmax(tpr_l >= 0.95)]
    p_in, r_in, _ = skm.precision_recall_curve(1 - ood, conf)
    p_out, r_out, _ = skm.precision_recall_curve(ood, -conf)
    return {"auroc": 100 * skm.auc(fpr_l, tpr_l), "fpr95": 100 * fpr95,
            "aupr_in": 100 * skm.auc(r_in, p_in), "aupr_out": 100 * skm.auc(r_out, p_out)}


def accuracy(bench, seed):
    d = features(bench, seed, f"test_{bench}")
    return 100 * float((d["logits"].argmax(1) == d["y"]).mean())
