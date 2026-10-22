Mean ± std over 3 training seeds (0, 1, 2). For each seed the metric is first averaged over the 7 OOD sets; the std is across seeds (population std). † = first 1000 images per set.

Test accuracy (posterior mean): 97.21 ± 0.09 %; 30-sample average: 97.20 ± 0.11 %.

**Per-image OOD detection, average over the 7 OOD sets**

| Approach | AUROC ↑ | FPR95 ↓ |
|---|---|---|
| MSP (max softmax) | 74.1 ± 4.8 | 74.1 ± 3.0 |
| Bayesian mutual information | 81.4 ± 4.8 | 60.2 ± 9.0 |
| k-NN distance † | 88.4 ± 0.9 | 46.4 ± 1.9 |
| [1] NP, per input (adaptation) | 55.4 ± 5.0 | 89.2 ± 4.9 |
| [2] TU (paper) | 79.1 ± 0.8 | 73.1 ± 3.1 |
| [2] TU, layers rescaled | 87.4 ± 0.6 | 60.1 ± 1.6 |
| [3] pTU (paper) | 83.0 ± 5.2 | 39.4 ± 7.0 |
| [4] PL full spectrum † | 82.6 ± 0.8 | 59.3 ± 1.5 |
| [4] β₀ only (PH part) † | 85.8 ± 0.1 | 50.3 ± 1.6 |
| [4] nonzero spectrum only † | 80.3 ± 1.2 | 62.5 ± 1.0 |
| [5] local ΔH0 + ΔH1 † | 94.7 ± 1.4 | 24.2 ± 6.3 |
| [5] local ΔH0 only † | 96.5 ± 0.4 | 16.9 ± 2.3 |

**Per-image AUROC per OOD set**

| Approach | FashionMNIST | KMNIST | CIFAR10 | SVHN | Gaussian | Uniform | MNIST-flip |
|---|---|---|---|---|---|---|---|
| MSP (max softmax) | 68.1 ± 4.4 | 84.1 ± 0.3 | 68.8 ± 6.2 | 64.9 ± 11.0 | 92.7 ± 0.7 | 59.8 ± 13.0 | 80.2 ± 1.2 |
| [2] TU (paper) | 82.2 ± 1.4 | 71.2 ± 1.7 | 80.6 ± 1.4 | 82.2 ± 0.7 | 95.3 ± 2.0 | 79.5 ± 7.9 | 62.6 ± 1.3 |
| [3] pTU (paper) | 82.1 ± 4.1 | 82.7 ± 4.9 | 94.5 ± 2.3 | 80.7 ± 4.8 | 46.9 ± 18.2 | 99.9 ± 0.1 | 94.5 ± 2.1 |
| [4] β₀ only (PH part) † | 84.6 ± 1.3 | 78.2 ± 1.7 | 77.8 ± 2.0 | 84.8 ± 2.3 | 98.2 ± 1.7 | 96.6 ± 0.5 | 80.7 ± 2.1 |
| [5] local ΔH0 only † | 93.8 ± 0.8 | 95.3 ± 0.3 | 99.2 ± 0.1 | 94.5 ± 1.0 | 93.5 ± 1.0 | 100.0 ± 0.0 | 99.3 ± 0.1 |
| k-NN distance † | 85.6 ± 1.9 | 85.4 ± 1.1 | 96.7 ± 0.5 | 85.4 ± 1.7 | 72.8 ± 1.6 | 99.9 ± 0.1 | 92.7 ± 0.4 |

**Batch-level OOD detection, average AUROC (FPR95) over the 7 OOD sets**

| Approach | m = 10 | m = 20 | m = 50 |
|---|---|---|---|
| MSP, batch mean | 85.6 ± 6.7 (43.2 ± 5.3) | 89.0 ± 8.3 (26.3 ± 8.0) | 92.0 ± 9.3 (13.7 ± 11.1) |
| [2] TU, batch mean | 94.6 ± 0.7 (21.8 ± 3.7) | 96.8 ± 0.5 (16.0 ± 3.2) | 98.9 ± 0.7 (5.4 ± 4.4) |
| [5] H0/H1 persistence + intrinsic dim. | 94.2 ± 0.5 (20.5 ± 1.2) | 98.9 ± 0.0 (4.8 ± 0.6) | 100.0 ± 0.0 (0.0 ± 0.0) |
| [6] SW distance to reference PDs | 92.1 ± 0.4 (24.5 ± 1.7) | 99.0 ± 0.2 (5.0 ± 0.7) | 100.0 ± 0.0 (0.0 ± 0.0) |
| [6] layer-to-layer SW profile | 90.2 ± 0.9 (40.5 ± 6.5) | 94.6 ± 1.6 (24.6 ± 8.3) | 97.3 ± 1.7 (10.1 ± 6.2) |

**[3] pTU permutation test (Algorithm 2): % of repeats rejecting H0 at α = 0.05**

| Test set | m = 20 | m = 50 | m = 100 |
|---|---|---|---|
| MNIST (ID) | 5 ± 2 | 3 ± 2 | 4 ± 2 |
| FashionMNIST | 99 ± 1 | 100 ± 0 | 100 ± 0 |
| KMNIST | 98 ± 2 | 100 ± 0 | 100 ± 0 |
| CIFAR10 | 100 ± 0 | 100 ± 0 | 100 ± 0 |
| SVHN | 100 ± 0 | 100 ± 0 | 100 ± 0 |
| Gaussian | 66 ± 19 | 96 ± 3 | 100 ± 0 |
| Uniform | 100 ± 0 | 100 ± 0 | 100 ± 0 |
| MNIST-flip | 100 ± 0 | 100 ± 0 | 100 ± 0 |
