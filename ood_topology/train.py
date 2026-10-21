"""Train the Bayes-by-Backprop MLP 784-64-16-10 on MNIST (pTU paper setup)."""
import argparse
import copy

import torch
import torch.nn.functional as F

from common import CACHE_DIR, BayesMLP, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--kl-scale", type=float, default=0.1,
                    help="multiplier on the KL term (1 = standard ELBO; 0.1 needed to reach the paper's 97.4%% MNIST accuracy)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    X, y = load_dataset("MNIST", "train")
    perm = torch.randperm(len(X))
    va, tr = perm[:5000], perm[5000:]
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    Xte, yte = load_dataset("MNIST", "test")

    model = BayesMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 128
    n_batches = (len(Xtr) + bs - 1) // bs
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(Xtr))
        tot = 0.0
        for b in range(n_batches):
            idx = order[b * bs:(b + 1) * bs]
            w = model.sample_weights()
            nll = F.cross_entropy(model(Xtr[idx], w), ytr[idx], reduction="sum")
            # Blundell et al.: KL spread uniformly over the minibatches of an epoch
            loss = (args.kl_scale * model.kl(w) / n_batches + nll) / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva, mean=True).argmax(1) == yva).float().mean().item()
        print(f"epoch {ep:2d}  loss {tot / n_batches:.4f}  val acc (posterior mean) {acc:.4f}", flush=True)
        if acc > best:
            best, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        acc_mean = (model(Xte, mean=True).argmax(1) == yte).float().mean().item()
        probs = torch.stack([F.softmax(model(Xte), 1) for _ in range(30)]).mean(0)
        acc_bma = (probs.argmax(1) == yte).float().mean().item()
    print(f"TEST acc: posterior mean {acc_mean:.4f} | 30-sample BMA {acc_bma:.4f}")
    torch.save(best_state, CACHE_DIR / f"bnn_seed{args.seed}{args.tag}.pt")
    (CACHE_DIR / f"bnn_seed{args.seed}{args.tag}_acc.txt").write_text(f"{acc_mean:.4f} {acc_bma:.4f}\n")


if __name__ == "__main__":
    main()
