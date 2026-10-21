"""Shared pieces: data, the Bayesian MLP used in the pTU paper, and a batched
maximum-spanning-tree routine for activation graphs (TU / pTU / NP)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
for d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

# Architecture used for all OOD experiments in Yeh & Yang (pTU), Sec. 5.2-5.4 / App.
ARCH = [784, 64, 16, 10]

OOD_SETS = ["FashionMNIST", "KMNIST", "CIFAR10", "SVHN", "Gaussian", "Uniform", "MNIST-flip"]


# --------------------------------------------------------------------------- data
def _to_tensor(ds, gray_resize=False):
    """Return an (N, 784) float tensor in [0, 1] (pTU preprocessing: grayscale, 28x28, [0,1])."""
    loader = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.reshape(len(x), -1))
        ys.append(torch.as_tensor(y))
    return torch.cat(xs), torch.cat(ys)


HF_TEST_FILES = {  # official Hugging Face copies of the test splits (much faster than the original hosts)
    "CIFAR10": ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
                "test-00000-of-00001.parquet", "img"),
    "SVHN": ("https://huggingface.co/datasets/ufldl-stanford/svhn/resolve/main/cropped_digits/"
             "test-00000-of-00001.parquet", "image"),
}


def _hf_parquet(name, transform):
    import io
    import urllib.request

    import pyarrow.parquet as pq
    from PIL import Image

    url, col = HF_TEST_FILES[name]
    f = DATA_DIR / "hf" / f"{name.lower()}_test.parquet"
    if not f.exists():
        f.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, f)
    t = pq.read_table(f).to_pydict()
    imgs = [transform(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t[col]]
    return list(zip(imgs, t["label"]))


def load_dataset(name, split="test", n=None, seed=0):
    """Load a dataset as (X[N,784] in [0,1], y[N]). Labels of OOD sets are irrelevant."""
    cache = CACHE_DIR / f"data_{name}_{split}.pt"
    if cache.exists():
        X, y = torch.load(cache)
    else:
        plain = T.ToTensor()
        gray = T.Compose([T.Grayscale(), T.Resize((28, 28), antialias=True), T.ToTensor()])
        train = split == "train"
        if name == "MNIST":
            ds = torchvision.datasets.MNIST(DATA_DIR, train=train, download=True, transform=plain)
        elif name == "FashionMNIST":
            ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=train, download=True, transform=plain)
        elif name == "KMNIST":
            ds = torchvision.datasets.KMNIST(DATA_DIR, train=train, download=True, transform=plain)
        elif name in ("CIFAR10", "SVHN") and split == "test":
            ds = _hf_parquet(name, gray)
        else:
            ds = None
        if ds is not None:
            X, y = _to_tensor(ds)
        elif name == "Gaussian":
            # TU paper: pixel values sampled from a Gaussian; matched to MNIST mean/std, clipped to [0,1]
            g = torch.Generator().manual_seed(123)
            X = (0.1307 + 0.3081 * torch.randn(10000, 784, generator=g)).clamp(0, 1)
            y = torch.zeros(10000, dtype=torch.long)
        elif name == "Uniform":
            g = torch.Generator().manual_seed(124)
            X = torch.rand(10000, 784, generator=g)
            y = torch.zeros(10000, dtype=torch.long)
        elif name == "MNIST-flip":
            # TU paper Sec. 4.3 corruption shift: switch n random pixels x -> 1 - x (here n = 100, ~13%)
            X, y = load_dataset("MNIST", "test")
            g = torch.Generator().manual_seed(125)
            X = X.clone()
            idx = torch.argsort(torch.rand(len(X), 784, generator=g), dim=1)[:, :100]
            rows = torch.arange(len(X)).unsqueeze(1)
            X[rows, idx] = 1 - X[rows, idx]
        else:
            raise ValueError(name)
        torch.save((X, y), cache)
    if n is not None and n < len(X):
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:n]
        X, y = X[perm], y[perm]
    return X.float(), y.long()


# --------------------------------------------------------------------------- model
class BayesLinear(nn.Module):
    """Bayes-by-Backprop linear layer (Blundell et al. 2015) with the scale-mixture prior
    used in the pTU paper (sigma1 = 1, sigma2 = 0.007, pi = 0.5)."""

    def __init__(self, d_in, d_out, pi=0.5, s1=1.0, s2=0.007):
        super().__init__()
        bound = 1 / math.sqrt(d_in)
        self.w_mu = nn.Parameter(torch.empty(d_out, d_in).uniform_(-bound, bound))
        self.w_rho = nn.Parameter(torch.full((d_out, d_in), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(d_out))
        self.b_rho = nn.Parameter(torch.full((d_out,), -5.0))
        self.pi, self.s1, self.s2 = pi, s1, s2

    def sample(self, mean=False):
        if mean:
            return self.w_mu, self.b_mu
        w = self.w_mu + F.softplus(self.w_rho) * torch.randn_like(self.w_mu)
        b = self.b_mu + F.softplus(self.b_rho) * torch.randn_like(self.b_mu)
        return w, b

    def log_prior(self, v):
        p1 = torch.distributions.Normal(0, self.s1).log_prob(v).exp()
        p2 = torch.distributions.Normal(0, self.s2).log_prob(v).exp()
        return torch.log(self.pi * p1 + (1 - self.pi) * p2 + 1e-30).sum()

    def log_q(self, v, mu, rho):
        return torch.distributions.Normal(mu, F.softplus(rho)).log_prob(v).sum()


class BayesMLP(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        self.layers = nn.ModuleList(BayesLinear(a, b) for a, b in zip(arch[:-1], arch[1:]))

    def sample_weights(self, mean=False):
        return [l.sample(mean) for l in self.layers]

    def forward(self, x, weights=None, mean=False):
        if weights is None:
            weights = self.sample_weights(mean)
        for i, (w, b) in enumerate(weights):
            x = F.linear(x, w, b)
            if i < len(weights) - 1:
                x = F.relu(x)
        return x

    def kl(self, weights):
        """Single-sample MC estimate of KL[q || prior] for the drawn weights."""
        kl = 0.0
        for l, (w, b) in zip(self.layers, weights):
            kl = kl + l.log_q(w, l.w_mu, l.w_rho) + l.log_q(b, l.b_mu, l.b_rho)
            kl = kl - l.log_prior(w) - l.log_prior(b)
        return kl


def forward_trace(weights, x):
    """Forward pass returning the inputs of every dense layer (x_0 = pixels, x_1, x_2 post-ReLU)
    and the logits. Activation graph of layer l uses |W_l(j,i) * x_{l}(i)|."""
    inputs = []
    h = x
    for i, (w, b) in enumerate(weights):
        inputs.append(h)
        h = F.linear(h, w, b)
        if i < len(weights) - 1:
            h = F.relu(h)
    return inputs, h


def load_model(seed, device="cpu"):
    m = BayesMLP()
    m.load_state_dict(torch.load(CACHE_DIR / f"bnn_seed{seed}.pt", map_location=device))
    return m.to(device).eval()


# --------------------------------------------------------------------------- topology core
@torch.no_grad()
def bipartite_mst_weights(M):
    """Maximum spanning tree of complete bipartite graphs, batched (Prim's algorithm).

    M: (B, n, m) non-negative edge weights between n "input" and m "output" neurons.
    Returns (B, n+m-1) MST edge weights sorted in decreasing order. Under the superlevel
    filtration these are exactly the death times of the 0-dim persistence diagram
    (all vertices born at +inf / 1), i.e. the PD used by NP [1], TU [2] and pTU [3].
    """
    B, n, m = M.shape
    dev = M.device
    neg = torch.tensor(-float("inf"), device=dev)
    ar = torch.arange(B, device=dev)
    key = torch.full((B, n + m), -float("inf"), device=dev)
    done = torch.zeros(B, n + m, dtype=torch.bool, device=dev)
    # start from input-neuron 0
    done[:, 0] = True
    key[:, n:] = M[:, 0, :]
    MT = M.transpose(1, 2).contiguous()
    out = torch.empty(B, n + m - 1, device=dev)
    for it in range(n + m - 1):
        k = torch.where(done, neg, key)
        val, idx = k.max(dim=1)
        out[:, it] = val
        done[ar, idx] = True
        is_in = idx < n
        i_in = idx.clamp(max=n - 1)
        j_out = (idx - n).clamp(min=0)
        row = M[ar, i_in]  # (B, m): edges from chosen input neuron to outputs
        col = MT[ar, j_out]  # (B, n): edges from chosen output neuron to inputs
        key[:, n:] = torch.where(is_in[:, None], torch.maximum(key[:, n:], row), key[:, n:])
        key[:, :n] = torch.where(~is_in[:, None], torch.maximum(key[:, :n], col), key[:, :n])
    return out.sort(dim=1, descending=True).values


def activation_graphs(weights, x):
    """Per-layer activation-graph weight tensors |W_l(j,i) x_l(i)| as (B, d_l, d_{l+1})."""
    inputs, logits = forward_trace(weights, x)
    graphs = [(h.abs().unsqueeze(2) * w.abs().t().unsqueeze(0)) for h, (w, _) in zip(inputs, weights)]
    return graphs, inputs, logits


def activation_pds(weights, x, chunk=1000):
    """0-dim PDs (sorted MST weights) of every layer's activation graph, plus logits."""
    pds = [[] for _ in weights]
    logits_all = []
    for s in range(0, len(x), chunk):
        graphs, _, logits = activation_graphs(weights, x[s:s + chunk])
        for l, G in enumerate(graphs):
            pds[l].append(bipartite_mst_weights(G).cpu())
        logits_all.append(logits.cpu())
    return [torch.cat(p) for p in pds], torch.cat(logits_all)


# --------------------------------------------------------------------------- evaluation helpers
N_EVAL = 2000     # samples per evaluation set (ID test and each OOD set)
N_TRAIN_REF = 10000  # training samples used to build reference statistics


def eval_sets():
    """Evaluation sets: 'MNIST' (in-distribution test) followed by the OOD sets."""
    return ["MNIST"] + OOD_SETS


def mean_net_pds(seed, name, split="test", n=N_EVAL):
    """Cached 0-dim activation-graph PDs + logits for the posterior-mean network."""
    f = CACHE_DIR / f"pd_mean_s{seed}_{name}_{split}_{n}.pt"
    if f.exists():
        return torch.load(f)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(seed, dev)
    X, _ = load_dataset(name, split, n=n)
    with torch.no_grad():
        weights = model.sample_weights(mean=True)
        pds, logits = activation_pds(weights, X.to(dev))
    out = {"pds": pds, "logits": logits}
    torch.save(out, f)
    return out


def ood_metrics(score_id, score_ood):
    """OOD is the positive class; higher score = more OOD.
    AUROC, and FPR95 = fraction of OOD samples accepted as ID at the threshold that
    accepts 95% of ID samples (Hendrycks & Gimpel convention)."""
    from sklearn.metrics import roc_auc_score
    s_id, s_ood = np.asarray(score_id, float), np.asarray(score_ood, float)
    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]
    auroc = roc_auc_score(y, np.r_[s_id, s_ood])
    thr = np.quantile(s_id, 0.95)
    fpr95 = float(np.mean(s_ood <= thr))
    return 100 * auroc, 100 * fpr95


def msp_score(logits):
    """Baseline (Hendrycks & Gimpel; the baseline used in the TU paper): 1 - max softmax."""
    return 1 - torch.softmax(torch.as_tensor(logits), 1).max(1).values.numpy()
