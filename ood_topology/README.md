# OOD detection with topology of neural-network layers

Implements the six approaches from `Topology_NN_Layers_Report` on **one shared model** and
evaluates each for out-of-distribution (OOD) detection. Methods are added one at a time.

## Shared model

Bayesian MLP **784 → 64 → 16 → 10** (ReLU) trained with Bayes-by-Backprop on MNIST: the exact
architecture and prior (scale mixture, σ₁ = 1, σ₂ = 0.007) of the pTU paper [3]. MLPs on MNIST
are also the setting of [1] and [2].

- Test accuracy: **97.1 %** (posterior mean), 97.1 % (30-sample average). The paper reports 97.39 %.
- The KL term is scaled by 0.1. With the full KL the model stalls at 94.5 %; the paper does not
  say how it weights the KL.
- Early stopping with patience 25, as in the paper. Weights: `cache/bnn_seed0.pt`.
- Deterministic methods ([1], [2], [4], [5], [6]) use the posterior-mean weights; pTU [3] uses
  posterior samples.

## Evaluation protocol

- In-distribution (ID): 2000 MNIST test images. Reference statistics: 10 000 MNIST train images.
- OOD sets (2000 each, all grayscale 28×28 in [0, 1], pTU preprocessing): Fashion-MNIST,
  KMNIST, CIFAR-10, SVHN, Gaussian noise, Uniform noise (as in [2]), and MNIST-flip (MNIST with
  100 random pixels flipped, x → 1 − x: the corruption shift of [2] §4.3).
- Metrics: AUROC (OOD = positive) and FPR95 = % of OOD accepted as ID at the threshold that
  accepts 95 % of ID samples. Baseline: MSP (1 − max softmax), the baseline used in [2].

## Results

### [1] Neural Persistence (Rieck et al., 2019): `m1_neural_persistence.py`

- **(a) Original NP** is computed from the weights alone (normalized by the global max weight,
  0-dim superlevel PD = MST, p = 2, normalized by the Theorem 1 bound, averaged over layers).
  Network NP = **0.716** (layers: 0.786 / 0.705 / 0.658). It does not depend on the input, so
  every sample gets the same score: **AUROC = 50 on every OOD set**. It cannot detect OOD.
- **(b) Input NP (our adaptation)** applies the same NP formula to each input's activation graph
  |W(j,i)·x(i)| (the graph of [2]). Score = |NP(x) − μₖ| / σₖ, using the training NP statistics
  of the predicted class k.

| OOD set | mean NP | NP AUROC ↑ | NP FPR95 ↓ | raw NP AUROC ↑ | MSP AUROC ↑ | MSP FPR95 ↓ |
|---|---|---|---|---|---|---|
| MNIST (ID) | 0.866 | – | – | – | – | – |
| Fashion-MNIST | 0.879 | 50.0 | 96.5 | 68.2 | 68.2 | 80.7 |
| KMNIST | 0.868 | 59.6 | 84.2 | 53.2 | 84.2 | 60.4 |
| CIFAR-10 | 0.877 | 48.0 | 95.4 | 65.0 | 62.5 | 84.9 |
| SVHN | 0.872 | 41.8 | 98.0 | 59.4 | 53.1 | 95.0 |
| Gaussian | 0.872 | 51.8 | 93.3 | 58.2 | 92.8 | 58.7 |
| Uniform | 0.874 | 40.4 | 99.2 | 61.1 | 43.1 | 99.6 |
| MNIST-flip | 0.869 | 57.0 | 89.4 | 54.2 | 80.2 | 67.5 |
| **average** | | **49.8** | **93.7** | **59.9** | **69.2** | **78.1** |

"raw NP" is a diagnostic one-sided score (higher NP = more OOD).

![NP histogram](results/m1_neural_persistence_s0.png)

**Takeaways for [1]:**
- NP was designed to describe a *trained network*, not an *input*. In its original form it
  carries no per-sample information.
- Applied per input, NP distributions of ID and OOD data overlap almost completely: OOD inputs
  have only slightly higher NP (0.868–0.879 vs 0.866). The class-conditional score is at chance
  (49.8), and the one-sided score reaches only 59.9, below the MSP baseline (69.2).
- Why: NP compresses each layer's whole PD into one normalized number. The global-max
  normalization also removes the activation scale, which is exactly where OOD inputs differ.
  This motivates [2], which keeps the full PD and compares it with class-wise reference PDs.
