#!/bin/zsh
# Phase 1: train the Bayesian MLP on several in-distribution sets and run all six methods.
# Usage: ./run_phase1.sh [ID ...]   (default: MNIST FashionMNIST KMNIST SVHN)
# Results: results/phase1/<ID>/, models and caches: cache/phase1/<ID>/
set -u
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
IDS=($@)
(( ${#IDS} )) || IDS=(MNIST FashionMNIST KMNIST SVHN)
SEEDS=(0 1 2)

for id in $IDS; do
  export OOD_ID=$id
  mkdir -p cache/phase1/$id
  # MNIST reuses the models of the original study (same training procedure)
  if [[ $id == MNIST ]]; then
    for s in $SEEDS; do cp -n cache/bnn_seed$s.pt cache/bnn_seed${s}_acc.txt cache/phase1/MNIST/ 2>/dev/null; done
  fi
  for s in $SEEDS; do
    [[ -f cache/phase1/$id/bnn_seed$s.pt ]] || python3 -u train.py --seed $s > cache/phase1/$id/train_seed$s.log 2>&1 &
  done
  wait
  echo "[$id] trained: $(cat cache/phase1/$id/bnn_seed*_acc.txt | tr '\n' ' ')"
  for s in $SEEDS; do
    for m in m1_neural_persistence m2_topological_uncertainty m3_ptu m4_persistent_laplacian m5_data_manifold m6_activation_ph; do
      python3 -u $m.py --seed $s > cache/phase1/$id/run_${m}_s$s.log 2>&1 || echo "[$id] FAILED $m seed $s"
    done
    echo "[$id] seed $s done"
  done
done
