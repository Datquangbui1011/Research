# Phase 1: several in-distribution datasets

Bayesian MLP 784-64-16-10 trained on each ID set; mean ± std over 3 seeds. Each cell first averages AUROC over the OOD sets of the group (near-OOD / far-OOD / covariate shift), then over seeds. † = first 1000 images per set.

## Test accuracy

| ID set | posterior mean | 30-sample average |
|---|---|---|
| MNIST | 97.21 ± 0.09 % | 97.20 ± 0.11 % |

## OOD sets per ID set

| ID set | near-OOD | far-OOD | covariate shift |
|---|---|---|---|
| MNIST | NotMNIST, FashionMNIST, KMNIST | CIFAR10, SVHN, Textures, TinyImageNet, Places365, Gaussian, Uniform | MNIST-flip, MNIST-C |

## ID = MNIST: per-image AUROC ↑ (FPR95 ↓)

| Method | near-OOD | far-OOD | cov. shift |
|---|---|---|---|
| MSP (max softmax) | 74.1 ± 2.6 (70.7 ± 1.2) | 69.2 ± 8.3 (80.3 ± 4.3) | 75.3 ± 1.5 (71.8 ± 1.3) |
| Bayesian mutual information | 80.9 ± 1.4 (56.4 ± 3.6) | 78.6 ± 8.4 (66.2 ± 11.5) | 79.3 ± 1.4 (61.8 ± 2.3) |
| k-NN distance † | 89.6 ± 0.9 (35.9 ± 2.0) | 90.6 ± 0.9 (32.3 ± 1.7) | 78.1 ± 0.3 (75.1 ± 1.6) |
| [1] NP, per input (adaptation) | 59.2 ± 3.0 (84.6 ± 3.9) | 53.5 ± 7.7 (91.5 ± 5.9) | 57.0 ± 1.1 (88.1 ± 1.7) |
| [2] TU (paper) | 80.7 ± 0.8 (63.9 ± 0.7) | 82.6 ± 1.7 (69.4 ± 2.9) | 65.3 ± 0.7 (88.2 ± 1.0) |
| [2] TU, layers rescaled | 85.5 ± 0.9 (55.9 ± 1.8) | 93.5 ± 0.1 (49.7 ± 2.5) | 71.6 ± 1.1 (85.0 ± 1.2) |
| [3] pTU (paper) | 87.0 ± 3.5 (32.3 ± 7.1) | 84.3 ± 5.0 (33.8 ± 6.4) | 75.0 ± 2.0 (53.5 ± 7.5) |
| [4] PL full spectrum † | 84.1 ± 0.6 (49.9 ± 2.3) | 83.1 ± 1.9 (54.4 ± 2.7) | 70.8 ± 1.6 (80.5 ± 0.9) |
| [4] β₀ only (PH part) † | 86.1 ± 0.7 (46.5 ± 2.8) | 84.6 ± 1.6 (47.2 ± 0.3) | 75.5 ± 1.0 (78.9 ± 2.9) |
| [5] local ΔH0 only † | 96.1 ± 0.3 (13.9 ± 1.2) | 97.0 ± 0.4 (14.8 ± 2.1) | 84.3 ± 0.2 (32.7 ± 0.5) |

Per OOD set (AUROC):

| Method | NotMNIST | FashionMNIST | KMNIST | CIFAR10 | SVHN | Textures | TinyImageNet | Places365 | Gaussian | Uniform | MNIST-flip | MNIST-C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSP (max softmax) | 70.2 ± 3.2 | 68.1 ± 4.4 | 84.1 ± 0.3 | 68.8 ± 6.2 | 64.9 ± 11.0 | 56.2 ± 15.0 | 70.4 ± 6.5 | 71.8 ± 6.8 | 92.7 ± 0.7 | 59.8 ± 13.0 | 80.2 ± 1.2 | 70.5 ± 1.9 |
| Bayesian mutual information | 80.0 ± 1.7 | 74.6 ± 2.8 | 88.3 ± 0.9 | 78.6 ± 5.7 | 75.4 ± 11.7 | 68.3 ± 16.3 | 79.4 ± 6.0 | 80.7 ± 5.8 | 93.8 ± 1.3 | 73.8 ± 12.7 | 85.6 ± 0.8 | 72.9 ± 1.9 |
| k-NN distance † | 97.8 ± 0.1 | 85.6 ± 1.9 | 85.4 ± 1.1 | 96.7 ± 0.5 | 85.4 ± 1.7 | 90.6 ± 1.4 | 95.1 ± 0.6 | 93.5 ± 0.7 | 72.8 ± 1.6 | 99.9 ± 0.1 | 92.7 ± 0.4 | 63.4 ± 0.2 |
| [1] NP, per input (adaptation) | 60.2 ± 2.8 | 57.5 ± 6.2 | 60.0 ± 0.7 | 55.3 ± 5.9 | 50.3 ± 10.0 | 48.4 ± 13.1 | 55.8 ± 6.2 | 56.9 ± 6.1 | 55.3 ± 2.6 | 52.4 ± 10.6 | 57.4 ± 0.3 | 56.6 ± 1.9 |
| [2] TU (paper) | 88.9 ± 0.9 | 82.2 ± 1.4 | 71.2 ± 1.7 | 80.6 ± 1.4 | 82.2 ± 0.7 | 82.5 ± 1.3 | 79.2 ± 1.0 | 79.1 ± 1.4 | 95.3 ± 2.0 | 79.5 ± 7.9 | 62.6 ± 1.3 | 68.0 ± 0.3 |
| [2] TU, layers rescaled | 93.7 ± 0.6 | 88.3 ± 1.3 | 74.4 ± 1.4 | 92.9 ± 0.4 | 94.1 ± 0.2 | 94.4 ± 0.1 | 91.7 ± 0.4 | 91.6 ± 0.4 | 94.9 ± 1.5 | 94.8 ± 1.3 | 72.1 ± 1.9 | 71.2 ± 0.4 |
| [3] pTU (paper) | 96.1 ± 1.6 | 82.1 ± 4.1 | 82.7 ± 4.9 | 94.5 ± 2.3 | 80.7 ± 4.8 | 86.7 ± 3.8 | 91.3 ± 2.9 | 90.3 ± 3.2 | 46.9 ± 18.2 | 99.9 ± 0.1 | 94.5 ± 2.1 | 55.5 ± 1.9 |
| [4] PL full spectrum † | 94.5 ± 0.5 | 82.5 ± 0.7 | 75.4 ± 2.1 | 76.7 ± 2.3 | 86.2 ± 2.7 | 83.6 ± 3.5 | 74.4 ± 2.2 | 74.6 ± 2.8 | 98.3 ± 1.2 | 88.2 ± 2.6 | 71.2 ± 2.9 | 70.3 ± 0.4 |
| [4] β₀ only (PH part) † | 95.7 ± 0.2 | 84.6 ± 1.3 | 78.2 ± 1.7 | 77.8 ± 2.0 | 84.8 ± 2.3 | 83.6 ± 3.0 | 75.3 ± 2.0 | 75.9 ± 2.3 | 98.2 ± 1.7 | 96.6 ± 0.5 | 80.7 ± 2.1 | 70.2 ± 0.4 |
| [5] local ΔH0 only † | 99.1 ± 0.1 | 93.8 ± 0.8 | 95.3 ± 0.3 | 99.2 ± 0.1 | 94.5 ± 1.0 | 95.7 ± 0.5 | 98.4 ± 0.2 | 97.8 ± 0.1 | 93.5 ± 1.0 | 100.0 ± 0.0 | 99.3 ± 0.1 | 69.3 ± 0.3 |

## Summary across ID sets: per-image AUROC (near / far / shift)

| Method | MNIST | mean over ID sets |
|---|---|---|
| MSP (max softmax) | 74.1 / 69.2 / 75.3 | **74.1** / **69.2** / **75.3** |
| Bayesian mutual information | 80.9 / 78.6 / 79.3 | **80.9** / **78.6** / **79.3** |
| k-NN distance † | 89.6 / 90.6 / 78.1 | **89.6** / **90.6** / **78.1** |
| [1] NP, per input (adaptation) | 59.2 / 53.5 / 57.0 | **59.2** / **53.5** / **57.0** |
| [2] TU (paper) | 80.7 / 82.6 / 65.3 | **80.7** / **82.6** / **65.3** |
| [2] TU, layers rescaled | 85.5 / 93.5 / 71.6 | **85.5** / **93.5** / **71.6** |
| [3] pTU (paper) | 87.0 / 84.3 / 75.0 | **87.0** / **84.3** / **75.0** |
| [4] PL full spectrum † | 84.1 / 83.1 / 70.8 | **84.1** / **83.1** / **70.8** |
| [4] β₀ only (PH part) † | 86.1 / 84.6 / 75.5 | **86.1** / **84.6** / **75.5** |
| [5] local ΔH0 only † | 96.1 / 97.0 / 84.3 | **96.1** / **97.0** / **84.3** |

## Batch level, m = 10 images: AUROC (near / far / shift)

| Method | MNIST |
|---|---|
| MSP, batch mean | 93.5 / 79.2 / 93.2 |
| [2] TU, batch mean | 98.1 / 98.6 / 84.3 |
| [5] H0/H1 persistence + intrinsic dim. | 87.4 / 98.9 / 82.9 |
| [6] SW distance to reference PDs | 83.1 / 98.0 / 82.4 |
| [6] layer-to-layer SW profile | 80.2 / 95.5 / 77.3 |

## Batch level, m = 20 images: AUROC (near / far / shift)

| Method | MNIST |
|---|---|
| MSP, batch mean | 97.6 / 83.4 / 97.7 |
| [2] TU, batch mean | 99.5 / 99.5 / 90.3 |
| [5] H0/H1 persistence + intrinsic dim. | 97.4 / 100.0 / 89.0 |
| [6] SW distance to reference PDs | 97.6 / 100.0 / 87.8 |
| [6] layer-to-layer SW profile | 92.0 / 98.7 / 78.9 |

## [3] pTU permutation test: % of repeats rejecting H0 (α = 0.05), m = 20 / 50

| ID set | ID test (false alarm) | near-OOD | far-OOD | cov. shift |
|---|---|---|---|---|
| MNIST | 5 / 3 | 99 / 100 | 95 / 100 | 73 / 97 |
