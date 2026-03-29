import os
import sys
sys.path.append('mdlm')
import json
import torch
import hydra
from tqdm import tqdm
import lightning

import smc.inference as inference


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(config):
    # Resolve relative checkpoint path (Hydra chdir breaks relative paths)
    if config.ft_model.ckpt_path and not os.path.isabs(config.ft_model.ckpt_path):
        config.ft_model.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ft_model.ckpt_path)

    # Set seed
    if config.run_all.seed is not None:
        lightning.seed_everything(config.run_all.seed)

    # Read all prompts from the prompt file
    with open(config.run_all.prompt_file, 'r') as f:
        prompts_from_file = [json.loads(l) for l in f]
        prompts_from_file = [p["context_string"] for p in prompts_from_file]

    all_text_samples = []
    for prompt_text in prompts_from_file:
        for _ in tqdm(list(range(config.run_all.runs_per_prompt))):
            config.smc.prompt_text = prompt_text
            print(f"Running SMC for prompt: {prompt_text}")
            text_samples, toxicity_scores = inference.main(config)
            if config.run_all.save_all:
                for txt, score in zip(text_samples, toxicity_scores):
                    all_text_samples.append({
                        "prompt": prompt_text,
                        "toxicity_score": score.item(),
                        "text": txt
                    })
            else:
                # Save the highest toxicity score and the corresponding text sample
                highest_toxicity_score = toxicity_scores.max()
                highest_index = toxicity_scores.argmax()
                highest_text_sample = text_samples[highest_index]
                all_text_samples.append({
                    "prompt": prompt_text,
                    "toxicity_score": highest_toxicity_score.item(),
                    "text": highest_text_sample
                })

    with open(f'text_samples.jsonl', 'w') as f:
        for sample in all_text_samples:
            f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    main()
