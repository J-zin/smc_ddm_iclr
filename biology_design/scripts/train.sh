#!/bin/bash
# Run from the biology_design/ directory.
# Hyperparameters used for the final paper results.

export LD_LIBRARY_PATH=

python train.py \
  finetuning.alpha=0.001 \
  finetuning.kl_weight=0.0 \
  finetuning.kl_method=forward \
  finetuning.entropy_weight=2.5
