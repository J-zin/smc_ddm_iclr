#!/bin/bash
# Evaluate a finetuned model using SMC.
# Run from the biology_design/ directory.
#
# Usage: bash scripts/eval.sh <ft_model_ckpt>
# Example: bash scripts/eval.sh model_weights/20260322/004927/ckpt_10/model.pth

export LD_LIBRARY_PATH=

ft_model_ckpt=$1

for n_particles in 1 2 4 8 16 32; do
    python eval.py \
        smc.num_particles=$n_particles \
        smc.final_strategy=argmax_rewards \
        smc.batch_p=8192 \
        smc.kl_weight=0.1 \
        smc.lambda_tempering.one_at=100 \
        smc.resampling.frequency=8 \
        smc.resampling.ess_threshold=null \
        smc.proposal_type=ft_model \
        smc.use_ft_model_for_expected_reward=True \
        smc.ft_model_ckpt="$ft_model_ckpt"
done
