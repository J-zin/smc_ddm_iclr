#!/bin/bash
# Run from the language_modelling/ directory.
# Hyperparameters used for the final paper results.

export LD_LIBRARY_PATH=

python train.py \
  finetuning.alpha=0.1 \
  finetuning.kl_weight=0.2 \
  finetuning.kl_method=backward
