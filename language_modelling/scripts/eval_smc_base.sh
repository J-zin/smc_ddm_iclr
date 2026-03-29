#!/bin/bash
# SMC with reverse (FK) proposal — base model, no finetuning.
# Run from the language_modelling/ directory.

export LD_LIBRARY_PATH=

runs_per_prompt=20

# Generate
for n_particles in 2 4 8 16; do
    python eval.py \
        run_all.runs_per_prompt=$runs_per_prompt \
        smc.proposal_type=reverse \
        smc.num_particles=$n_particles \
        smc.batch_p=$n_particles \
        loader.eval_batch_size=$n_particles \
        smc.kl_weight=0.1 \
        smc.lambda_tempering.enabled=False \
        smc.resampling.frequency=20 \
        smc.phi=4
done

# Convert to evaluation format
python evaluation/mdlm_to_eval_format.py \
    --glob_expression "outputs/reverse_*/*/*/text_samples.jsonl" \
    --expected_per $runs_per_prompt \
    --prompt_path evaluation/pplm_discrim_prompts_orig.jsonl \
    --max_len 1000

# Compute metrics
for path in outputs/reverse_*/*/*/*_gen.jsonl; do
    echo "$path"
    python evaluation/evaluate.py \
        --generations_file "$path" \
        --metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext \
        --output_file "$(basename "$path")_eval.txt"
done
