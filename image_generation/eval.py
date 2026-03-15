import os, sys
import hydra
from omegaconf import OmegaConf
from collections import defaultdict
from functools import partial
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import wandb
from functools import partial
import logging
import math
from copy import deepcopy
import json

from pathlib import Path

from meissonic.smc.transformer import Transformer2DModel
from meissonic.smc.smc_pipeline import Pipeline
from meissonic.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module

from peft import LoraConfig

import numpy as np
import torch
import torch.utils.checkpoint
import torch.distributed as dist
from utils.distributed import init_distributed_singlenode, set_seed, setup_for_distributed

import alignment.prompts
import alignment.rewards

from meissonic.smc.resampling import resample
from meissonic.smc.scheduler import MeissonicScheduler

def load_pipeline(device, dtype):
    model_path = "Collov-Labs/Monetico"
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype) # better for Monetico
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)
    scheduler_new = MeissonicScheduler(
        mask_token_id=scheduler.config.mask_token_id, # type: ignore
        masking_schedule=scheduler.config.masking_schedule, #  type: ignore
    )
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler_new)
    pipe.to(device)
    return pipe

def load_prompts(prompt_name):
    from alignment.prompts import ASSETS_PATH
    from alignment.prompts import _load_lines, read_csv, read_hpd
    if prompt_name == "simple_animals":
        prompts = _load_lines("simple_animals.txt")
    elif prompt_name == "drawbench":
        prompts = read_csv(f"alignment/assets/DrawBench Prompts.csv")
    elif prompt_name == "hpd_photo_painting":
        prompts = []
        with open(ASSETS_PATH.joinpath(f"HPDv2/benchmark_photo.json"), "r") as f:
            prompts.extend(json.load(f)[:10]) # 790 for train, 10 for test
        with open(ASSETS_PATH.joinpath(f"HPDv2/benchmark_paintings.json"), "r") as f:
            prompts.extend(json.load(f)[:10]) # 790 for train, 10 for test
    else:
        raise NotImplementedError(f"Prompt {prompt_name} not implemented")
    return prompts

@hydra.main(version_base=None, config_path="config", config_name="monetico")
def main(config):
    logging.basicConfig(
        # filename='output/app.log',    # Uncomment to log to a file
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    # make the Hydra config object accept attributes that aren't defined in the config file
    OmegaConf.set_struct(config, False)

    config['gpu_type'] = torch.cuda.get_device_name() \
                            if torch.cuda.is_available() else "CPU"
    logger.info(f"GPU type: {config['gpu_type']}")

    name = f"klreg{config['train']['klreg']}_rewardexp{config['train']['reward_exp']}_nrewardest{config['train']['num_rewardest_samples']}_tmp{config['sample']['softmax_temperature']}_{config['prompt_fn']}_{config['reward_fn']}"
    output_dir = os.path.join(config['output_dir'], name)
    wandb_name = f"{config['smc']['proposal_type']}_nparticles{config['smc']['num_particles']}_nsteps{config['smc']['num_inference_steps']}_klreg{config['train']['klreg']}_{config['prompt_fn']}_{config['reward_fn']}"
    if config['wandb']:
        wandb.init(
            project="eval ft Meissonic", 
            config=OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            ),
            name=f"lora_{wandb_name}" if config['use_lora'] else wandb_name,
            mode=config['wandb_mode'] if is_local_main_process else "disabled")

    logger.info(f"\n{config}")
    set_seed(config['seed'])

    weight_dtype = torch.float32
    if config['mixed_precision'] == "fp16":
        weight_dtype = torch.float16
    elif config['mixed_precision'] == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)

    pipeline = load_pipeline(device, weight_dtype)
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    if config['use_lora']:
        transformer = pipeline.transformer
        # transformer_lora_config = LoraConfig(
        #     r=config['train']['lora_rank'], lora_alpha=config['train']['lora_rank'],
        #     init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        # )
        # transformer.add_adapter(transformer_lora_config)
        if config['mixed_precision'] in ["fp16", "bf16"]:
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(transformer, dtype=torch.float32)

        save_path = os.path.join(output_dir, f"checkpoint_epoch175")
        pipeline.load_lora_weights(
            pretrained_model_name_or_path_or_dict=save_path,
        )

    prompts = load_prompts(config['prompt_fn'])
    reward_fn = getattr(alignment.rewards, config['reward_fn'])(weight_dtype, device)
    differentiable_reward_fn = getattr(
        alignment.rewards, f"differentiable_{config['reward_fn']}"
    )(weight_dtype, device) if config['smc']['proposal_type'] == "locally_optimal" else reward_fn

    lambda_one_at = steps = config['smc']['num_inference_steps']
    lambdas = torch.cat([torch.linspace(0, 1, lambda_one_at + 1), torch.ones(steps - lambda_one_at)])

    batch_size = config['smc']['batch_size']
    assert batch_size == 1
    num_batches = math.ceil(len(prompts) / (batch_size * world_size))
    logger.info(f"Evaluating {len(prompts)} prompts in {num_batches} batches of size {batch_size}x{world_size}={batch_size*world_size}")
    all_scores = []
    for i in tqdm(range(num_batches), position=0, desc="Eval batches", disable=not is_local_main_process):
        batch_prompts = []
        for j in range(batch_size):
            idx = (i * batch_size * world_size + local_rank * batch_size + j) % len(prompts)
            prompt = prompts[idx]
            batch_prompts.append(prompt)
            
        image_reward_fn = lambda images: differentiable_reward_fn(images, [batch_prompts[0]] * len(images), None)[0]
        images = pipeline(
            prompt=batch_prompts[0], 
            reward_fn=image_reward_fn,
            resample_fn=lambda log_w: resample(log_w, ess_threshold=0.5, partial=True),
            resample_frequency=config['smc']['resample_frequency'],
            negative_prompt=config['negative_prompt'],
            height=512,
            width=512,
            guidance_scale=config['sample']['guidance_scale'],
            num_inference_steps=config['smc']['num_inference_steps'],
            kl_weight=config['smc']['kl_weight'],
            lambdas=lambdas,
            num_particles=config['smc']['num_particles'],
            batch_p=config['smc']['batch_p'],
            proposal_type=config['smc']['proposal_type'],
            use_continuous_formulation=True,
            phi=int(config['smc']['phi']),
            tau=config['smc']['tau'],
            output_type="pt",
            verbose=False
        )
        rewards, _ = reward_fn(images, batch_prompts * len(images), None)
        best_reward, best_reward_idx = rewards.max(dim=0)
        all_scores.append(best_reward.item())
        del images, rewards, best_reward
        torch.cuda.empty_cache()

        if is_local_main_process:
            wandb.log({
                'itrs': i,
                'total_itrs': num_batches
            })
            
    # gather all scores to main process
    all_scores = torch.tensor(all_scores, device=device)
    gather_list = [torch.zeros_like(all_scores) for _ in range(world_size)]
    dist.all_gather(gather_list, all_scores)
    if is_local_main_process:
        all_scores = torch.cat(gather_list, dim=0).cpu().numpy().tolist()
        avg_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        logger.info(f"Avg reward: {avg_score:.8f} +/- {std_score:.8f}")

        wandb.log({
            "avg_reward_mean": avg_score,
            "avg_reward_std": std_score,
        })

if __name__ == "__main__":
    main()
    dist.destroy_process_group()