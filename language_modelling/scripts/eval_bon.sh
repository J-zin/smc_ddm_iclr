#!/bin/bash
# Best-of-N baseline (no SMC, just sample N sequences and pick best).
# Run from the language_modelling/ directory.

export LD_LIBRARY_PATH=

runs_per_prompt=20

# Generate
for n_particles in 1 2 4 8 16; do
    python eval.py \
        run_all.runs_per_prompt=$runs_per_prompt \
        smc.proposal_type=without_SMC \
        smc.num_particles=$n_particles \
        smc.batch_p=$n_particles \
        loader.eval_batch_size=$n_particles
done

# Convert to evaluation format
python evaluation/mdlm_to_eval_format.py \
    --glob_expression "outputs/without_SMC_*/*/*/text_samples.jsonl" \
    --expected_per $runs_per_prompt \
    --prompt_path evaluation/pplm_discrim_prompts_orig.jsonl \
    --max_len 1000

# Compute metrics
for path in outputs/without_SMC_*/*/*/*_gen.jsonl; do
    echo "$path"
    python evaluation/evaluate.py \
        --generations_file "$path" \
        --metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext \
        --output_file "$(basename "$path")_eval.txt"
done
