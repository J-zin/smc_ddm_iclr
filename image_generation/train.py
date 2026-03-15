import os, sys
import hydra
from omegaconf import OmegaConf
from collections import defaultdict
import contextlib
import datetime
import time
from concurrent import futures
import wandb
from functools import partial
import tempfile
from PIL import Image
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import logging
import math
import pickle, gzip
from copy import deepcopy

from meissonic.transformer import Transformer2DModel
from meissonic.pipeline import Pipeline
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
from peft.utils import get_peft_model_state_dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import init_distributed_singlenode, set_seed, setup_for_distributed

import alignment.prompts
import alignment.rewards
from alignment.f_psi import FpsiModel
from alignment.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from alignment.diffusers_patch.confidence_with_logprob import log_prob_diff_step, pred_orig_latent

from meissonic.pipeline import _prepare_latent_image_ids

def load_pipeline(device, dtype):
    model_path = "Collov-Labs/Monetico"
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype) # better for Monetico
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=dtype)
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
    pipe.to(device)
    return pipe

def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model

@torch.no_grad()
def estimate_reward(reward_fn, images, prompts, prompts_metadata, num_samples=4):
    """
    images: (bs*num_samples, 3, 32, 32), bf16 in [0, 1]
    prompts: list of strings
    prompts_metadata: list of dicts
    """
    prompts = [p for p in prompts for _ in range(num_samples)]
    prompts_metadata = [m for m in prompts_metadata for _ in range(num_samples)]
    logr = reward_fn(images, prompts, prompts_metadata)[0]  # (bs * num_samples,)
    logr = logr.reshape(-1, num_samples)  # (bs, num_samples)
    logr = logr.logsumexp(dim=1) - math.log(num_samples)  # (bs,)
    return logr

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

    name = f"klreg{config['train']['klreg']}_rewardexp{config['train']['reward_exp']}_nrewardest{config['train']['num_rewardest_samples']}_tmp{config['sample']['softmax_temperature']}_cfg{config['sample']['guidance_scale']}_{config['prompt_fn']}_{config['reward_fn']}"
    output_dir = os.path.join(config['output_dir'], name)
    os.makedirs(output_dir, exist_ok=True)
    if config['wandb']:
        wandb.init(
            project="logvar-alignment Meissonic", 
            config=OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            ),
            name=name,
           save_code=True, mode=config['wandb_mode'] if is_local_main_process else "disabled")

    logger.info(f"\n{config}")
    set_seed(config['seed'])

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if config['mixed_precision'] == "fp16":
        weight_dtype = torch.float16
    elif config['mixed_precision'] == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)

    pipeline = load_pipeline(device, weight_dtype)
    pipeline.vqvae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    transformer = pipeline.transformer
    transformer.requires_grad_(False)
    for param in transformer.parameters():
        param.requires_grad_(False)

    # transformer_ref = deepcopy(transformer)
    # for param in transformer_ref.parameters():
    #     assert not param.requires_grad
    # transformer_ref.eval()

    assert config['use_lora']
    transformer.to(device, dtype=weight_dtype)
    transformer_lora_config = LoraConfig(
        r=config['train']['lora_rank'], lora_alpha=config['train']['lora_rank'],
        init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    if config['mixed_precision'] in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)
    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    """ load LoRA weights if necessary
    save_path = os.path.join(output_dir, f"checkpoint_debug")
    pipeline.load_lora_weights(
        pretrained_model_name_or_path_or_dict=save_path,
    )
    """

    scaler = None
    if config['mixed_precision'] in ["fp16", "bf16"]:
        scaler = torch.cuda.amp.GradScaler()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 is True by default
        torch.backends.cudnn.benchmark = True

    if config['train']['use_8bit_adam']:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # prepare prompt and reward fn
    prompt_fn = getattr(alignment.prompts, config['prompt_fn'])
    reward_fn = getattr(alignment.rewards, config['reward_fn'])(weight_dtype, device)

    # generate negative prompt embeddings
    neg_prompt_outputs = pipeline.text_encoder(
        pipeline.tokenizer(
            [f"{config['negative_prompt']}"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length, # 77
        ).input_ids.to(device),
        return_dict=True,
        output_hidden_states=True
    )
    negative_prompt_embed = neg_prompt_outputs.text_embeds # (1, 1024)
    negative_encoder_hidden_states = neg_prompt_outputs.hidden_states[-2] # (1, 77, 1024)

    sample_neg_prompt_embeds = negative_prompt_embed.repeat(config['sample']['batch_size'], 1)
    train_neg_prompt_embeds = negative_prompt_embed.repeat(config['train']['batch_size'], 1)
    sample_neg_encoder_hidden_states = negative_encoder_hidden_states.repeat(config['sample']['batch_size'], 1, 1)
    train_neg_encoder_hidden_states = negative_encoder_hidden_states.repeat(config['train']['batch_size'], 1, 1)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    def func_autocast():
        return torch.cuda.amp.autocast(dtype=weight_dtype)
    if config['use_lora']:
        # LoRA weights are actually float32, but other part of Meissonic are in bf16/fp16
        autocast = contextlib.nullcontext
    else:
        autocast = func_autocast

    transformer.to(device)
    transformer = DDP(transformer, device_ids=[local_rank])

    #######################################################
    ################ FOR LogVar Loss ######################
    def decode(latents):
        needs_upcasting = pipeline.vqvae.dtype == torch.float16 and pipeline.vqvae.config.force_upcast
        if needs_upcasting: pipeline.vqvae.float()
        image = pipeline.vqvae.decode(
            latents,
            force_not_quantize=True,
            shape=(
                latents.shape[0],
                512 // pipeline.vae_scale_factor,
                512 // pipeline.vae_scale_factor,
                pipeline.vqvae.config.latent_channels,
            ),
        ).sample.clip(0, 1)
        image = pipeline.image_processor.postprocess(image, output_type='pt')
        if needs_upcasting: pipeline.vqvae.half()
        return image

    f_psi = FpsiModel()
    f_psi = f_psi.to(device, dtype=torch.float32)
    f_psi = DDP(f_psi, device_ids=[local_rank])
    
    params = [
        {"params": lora_layers, "lr": config['train']['learning_rate']},
        {"params": f_psi.parameters(), "lr": config['train']['learning_rate']}
    ]
    optimizer = optimizer_cls(
        params,
        betas=(config['train']['adam_beta1'], config['train']['adam_beta2']),
        weight_decay=config['train']['adam_weight_decay'],
        eps=config['train']['adam_epsilon'],
    )

    result = defaultdict(dict)
    result["config"] = config
    start_time = time.time()

    #######################################################
    # Start!
    samples_per_epoch = (
        config['sample']['batch_size'] * num_processes
        * config['sample']['num_batches_per_epoch']
    )
    total_train_batch_size = (
        config['train']['batch_size'] * num_processes
        * config['train']['gradient_accumulation_steps']
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config['num_epochs']}")
    logger.info(f"  Sample batch size per device = {config['sample']['batch_size']}")
    logger.info(f"  Train batch size per device = {config['train']['batch_size']}")
    logger.info(
        f"  Gradient Accumulation steps = {config['train']['gradient_accumulation_steps']}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = test_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config['train']['num_inner_epochs']}")

    assert config['sample']['batch_size'] >= config['train']['batch_size']
    assert config['sample']['batch_size'] % config['train']['batch_size'] == 0 # not necessary
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = 0
    global_step = 0
    for epoch in range(first_epoch, config['num_epochs']):
        if config['train']['anneal'] in ["linear"]:
            ratio = min(1, epoch / 50.)  # warm up for 50 epochs
        else:
            ratio = 1.
        reward_exp_ep = config['train']['reward_exp'] * ratio
        def reward_transform(value):
            return value * reward_exp_ep
        
        num_diffusion_steps = config['sample']['num_steps']
        num_train_timesteps = int(num_diffusion_steps * config['train']['timestep_fraction'])
        accumulation_steps = config['train']['gradient_accumulation_steps'] * num_train_timesteps

        #################### SAMPLING ####################
        torch.cuda.empty_cache()
        transformer.zero_grad()
        transformer.eval()
        f_psi.zero_grad()

        if True:
            with torch.inference_mode(): # similar to torch.no_grad() but also disables autograd.grad()
                samples = []
                prompts = []

                for i in tqdm(
                    range(config['sample']['num_batches_per_epoch']),
                    desc=f"Epoch {epoch}: sampling",
                    disable=not is_local_main_process,
                    position=0,
                ):
                    # generate prompts
                    prompts, prompt_metadata = zip(
                        *[
                            prompt_fn(**config['prompt_fn_kwargs'])
                            for _ in range(config['sample']['batch_size'])
                        ]
                    )

                    # encode prompts
                    prompt_ids = pipeline.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=pipeline.tokenizer.model_max_length, # 77,
                    ).input_ids.to(device)
                    prompt_outputs = pipeline.text_encoder(
                        prompt_ids, return_dict=True, output_hidden_states=True)
                    prompt_embeds = prompt_outputs.text_embeds
                    encoder_hidden_states = prompt_outputs.hidden_states[-2]
                    
                    # sample
                    with autocast():
                        ret_tuple = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_encoder_hidden_states=sample_neg_encoder_hidden_states,
                            num_inference_steps=num_diffusion_steps,
                            guidance_scale=config['sample']['guidance_scale'],
                            softmax_temperature=config['sample']['softmax_temperature'],
                            output_type="pt",
                        )
                    images, latents = ret_tuple
                    
                    latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 32, 32)
                    timesteps = pipeline.scheduler.timesteps.repeat(
                        config['sample']['batch_size'], 1
                    )  # (bs, num_steps)  (47, ..., 1, 0) corresponds to "next_latents"

                    rewards = reward_fn(images, prompts, prompt_metadata) # (reward, reward_metadata)

                    samples.append(
                        {
                            "prompts": prompts, # tuple of strings
                            "prompt_metadata": prompt_metadata,

                            "prompt_ids": prompt_ids,
                            "prompt_embeds": prompt_embeds,
                            "encoder_hidden_states": encoder_hidden_states,
                            "timesteps": timesteps,
                            "latents": latents[
                                :, :-1
                            ],  # each entry is the latent before timestep t
                            "next_latents": latents[
                                :, 1:
                            ],  # each entry is the latent after timestep t
                            "rewards": rewards,
                        }
                    )

                # wait for all rewards to be computed
                for sample in tqdm(
                    samples,
                    desc="Waiting for rewards",
                    disable=not is_local_main_process,
                    position=0,
                ):
                    rewards, reward_metadata = sample["rewards"]
                    sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            new_samples = {}
            for k in samples[0].keys():
                if k in ["prompts", "prompt_metadata"]:
                    # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                    new_samples[k] = [item for s in samples for item in s[k]]
                else:
                    new_samples[k] = torch.cat([s[k] for s in samples])
            samples = new_samples

            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images):
                    # bf16 cannot be converted to numpy directly
                    pil = Image.fromarray(
                        (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                    # pil.save(os.path.join('./output/images', f"{prompts[i]}.jpg"))
                if config['wandb'] and is_local_main_process:
                    wandb.log(
                        {
                            "images": [
                                wandb.Image(
                                    os.path.join(tmpdir, f"{i}.jpg"),
                                    caption=f"{prompt} | {reward:.2f}",
                                )
                                for i, (prompt, reward) in enumerate(
                                    zip(prompts, rewards)
                                )
                            ],
                        },
                        step=global_step,
                    )

            rewards = torch.zeros(world_size * len(samples["rewards"]),
                          dtype=samples["rewards"].dtype, device=device)
            dist.all_gather_into_tensor(rewards, samples["rewards"])
            rewards = rewards.cpu().float().numpy()
            result["reward_mean"][global_step] = rewards.mean()
            result["reward_std"][global_step] = rewards.std()

            if is_local_main_process:
                logger.info(f"global_step: {global_step}  rewards: {rewards.mean().item():.3f}")
                if config['wandb']:
                    wandb.log(
                        {
                            "reward_mean": rewards.mean(), # samples["rewards"].mean()
                            "reward_std": rewards.std(),
                        },
                        step=global_step,
                    )

            del samples["prompt_ids"]

            total_batch_size, num_timesteps = samples["timesteps"].shape
            assert (
                total_batch_size
                == config['sample']['batch_size'] * config['sample']['num_batches_per_epoch']
            )
            assert num_timesteps == num_diffusion_steps

        #################### TRAINING ####################
        for inner_epoch in range(config['train']['num_inner_epochs']):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=device)
            for k, v in samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples[k] = [v[i] for i in perm]
                elif k in ["tf_outputs"]:
                    samples[k] = v[perm.cpu()]  # we put tf_outputs in cpu to save gpu memory
                else:
                    samples[k] = v[perm]
            
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=device)
                    for _ in range(total_batch_size)
                ]
            ) # (total_batch_size, num_steps)
            # "prompts" & "prompt_metadata" are constant along time dimension
            key_ls = ["timesteps", "latents", "next_latents"]
            for key in key_ls:
                samples[key] = samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]

            ### rebatch for training
            samples_batched = {}
            for k, v in samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples_batched[k] = [v[i:i + config['train']['batch_size']]
                                for i in range(0, len(v), config['train']['batch_size'])]
                elif k in ["tf_outputs"]:
                    samples_batched[k] = v.reshape(-1, config['train']['batch_size'], *v.shape[1:])
                else:
                    samples_batched[k] = v.reshape(-1, config['train']['batch_size'], *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ] # len = sample_bs * num_batches_per_epoch // train_bs = num_train_batches_per_epoch

            transformer.train()
            f_psi.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_local_main_process,
            ):
                """
                sample: [
                ('prompts', list of strings, len=train_bs), ('prompt_metadata', list of dicts),
                (bf16) ('prompt_embeds', torch.Size([train_bs, 1024])), 
                (bf16) ('encoder_hidden_states', torch.Size([train_bs, 77, 1024])),
                (int64) ('timesteps', torch.Size([train_bs, 48])), 
                (bf16) ('latents', torch.Size([train_bs, 48, 32, 32])), ('next_latents', torch.Size([train_bs, 48, 32, 32])), 
                (float32) ('rewards', torch.Size([train_bs])
                # if config['train']['klreg'] > 0:
                #     (bf16) ('tf_outputs', torch.Size([train_bs, 48, 8192, 32, 32])
                ]
                """
                if config['train']['cfg']:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                    hidden_states = torch.cat(
                        [train_neg_encoder_hidden_states, sample["encoder_hidden_states"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    hidden_states = sample["encoder_hidden_states"]

                micro_conds = torch.tensor(
                    [512, 512, 0, 0, 6],
                    device=device,
                    dtype=encoder_hidden_states.dtype,
                )
                micro_conds = micro_conds.unsqueeze(0)
                micro_conds = micro_conds.expand(embeds.shape[0], -1)

                for j in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not is_local_main_process):
                    model_input = sample["latents"][:, j]
                    img_ids = _prepare_latent_image_ids(model_input.shape[0], 2*model_input.shape[-2], 2*model_input.shape[-1],model_input.device, model_input.dtype)
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                    
                    with autocast():
                        if config['train']['cfg']:
                            model_output = transformer(
                                hidden_states=torch.cat([model_input] * 2),
                                micro_conds=micro_conds,
                                pooled_projections=embeds,
                                encoder_hidden_states=hidden_states,
                                img_ids = img_ids,
                                txt_ids = txt_ids,
                                timestep=torch.cat([sample["timesteps"][:, j]] * 2),
                            )
                            uncond_logits, cond_logits = model_output.chunk(2)
                            model_output = (
                                uncond_logits 
                                + config['sample']['guidance_scale'] 
                                * (cond_logits - uncond_logits)
                            )
                            
                        else:
                            model_output = transformer(
                                hidden_states=model_input,
                                micro_conds=micro_conds,
                                pooled_projections=embeds,
                                encoder_hidden_states=hidden_states,
                                img_ids = img_ids,
                                txt_ids = txt_ids,
                                timestep=sample["timesteps"][:, j],
                            )   # (bs, 8192, 32, 32)
                            
                    with autocast(), torch.no_grad():
                        transformer.module.disable_adapters()
                        if config['train']['cfg']:
                            ref_model_output = transformer(
                                hidden_states=torch.cat([model_input] * 2),
                                micro_conds=micro_conds,
                                pooled_projections=embeds,
                                encoder_hidden_states=hidden_states,
                                img_ids = img_ids,
                                txt_ids = txt_ids,
                                timestep=torch.cat([sample["timesteps"][:, j]] * 2),
                            )
                            uncond_logits, cond_logits = ref_model_output.chunk(2)
                            ref_model_output = (
                                uncond_logits 
                                + config['sample']['guidance_scale'] 
                                * (cond_logits - uncond_logits)
                            )

                        else:
                            ref_model_output = transformer(
                                hidden_states=model_input,
                                micro_conds=micro_conds,
                                pooled_projections=embeds,
                                encoder_hidden_states=hidden_states,
                                img_ids = img_ids,
                                txt_ids = txt_ids,
                                timestep=sample["timesteps"][:, j],
                            )   # (bs, 8192, 32, 32)
                        transformer.module.enable_adapters()

                    #######################################################
                    ################# Compute LogVar Loss #################
                    #######################################################

                    log_q_psi = F.log_softmax(model_output, dim=1)
                    log_p_ref = F.log_softmax(ref_model_output, dim=1)

                    log_prob_diff = log_prob_diff_step(
                        pipeline, log_q_psi, log_p_ref, sample["latents"][:, j], sample["next_latents"][:, j]
                    )
                    info["log_prob_diff"].append(torch.mean(log_prob_diff).detach())
                    
                    if config['train']['klreg'] > 0:
                        assert model_output.requires_grad
                        assert not ref_model_output.requires_grad
                        # compute reversed KL divergence KL(q_psi || p_ref)
                        klreg = F.kl_div(log_p_ref, log_q_psi, reduction='none', log_target=True).sum(dim=(1, 2, 3))

                    with autocast(), torch.no_grad():
                        # we use transformer here rather than transformer_ref, 
                        #   this will change the intermediate target distribution,
                        #   but in theory the optimal at t=0 remains the same
                        model_output = transformer(
                            hidden_states=sample["latents"][:, j],
                            micro_conds=micro_conds[:sample["prompt_embeds"].shape[0], ...],
                            pooled_projections=sample["prompt_embeds"],
                            encoder_hidden_states=sample['encoder_hidden_states'],
                            img_ids = img_ids,
                            txt_ids = txt_ids,
                            timestep=sample["timesteps"][:, j],
                        )   # (bs, 8192, 32, 32)
                        latent = pred_orig_latent(pipeline.scheduler, model_output, sample["latents"][:, j], config['train']['num_rewardest_samples'])
                    with torch.inference_mode():
                        latent = latent.reshape(-1, *latent.shape[2:])  # (bs * num_rewardest_samples, 32, 32)
                        logr_tmp = estimate_reward(
                            reward_fn, decode(latent), sample["prompts"], sample["prompt_metadata"], config['train']['num_rewardest_samples']
                        )
                    logr = reward_transform(logr_tmp)   # float32

                    with autocast(), torch.no_grad():
                        # we use transformer here rather than transformer_ref, 
                        #   this will change the intermediate target distribution,
                        #   but in theory the optimal at t=0 remains the same
                        timestep_next =  torch.clamp(sample["timesteps"][:, j] - 1, min=0)
                        model_output = transformer(
                            hidden_states=sample["next_latents"][:, j],
                            micro_conds=micro_conds[:sample["prompt_embeds"].shape[0], ...],
                            pooled_projections=sample["prompt_embeds"],
                            encoder_hidden_states=sample['encoder_hidden_states'],
                            img_ids = img_ids,
                            txt_ids = txt_ids,
                            timestep=timestep_next,
                        )   # (bs, 8192, 32, 32)
                        latent_next = pred_orig_latent(pipeline.scheduler, model_output, sample["next_latents"][:, j], config['train']['num_rewardest_samples'])
                    with torch.inference_mode():
                        latent_next = latent_next.reshape(-1, *latent_next.shape[2:])  # (bs * num_rewardest_samples, 32, 32)
                        logr_next_tmp = estimate_reward(
                            reward_fn, decode(latent_next), sample["prompts"], sample["prompt_metadata"], config['train']['num_rewardest_samples']
                        )
                    logr_next = reward_transform(logr_next_tmp) # float32
                    end_mask = sample["timesteps"][:, j] == pipeline.scheduler.timesteps[-1] # RHS is 0
                    logr_next[end_mask] = reward_transform(sample['rewards'][end_mask].to(logr_next))

                    log_reward_diff = logr_next - logr
                    info["log_reward_diff"].append(torch.mean(log_reward_diff).detach())

                    log_w = log_reward_diff + log_prob_diff
                    F_t = f_psi(
                        timesteps=sample["timesteps"][:, j], 
                        micro_conds=micro_conds[:sample["prompt_embeds"].shape[0], ...],
                        pooled_projections=sample["prompt_embeds"], 
                        encoder_hidden_states=sample["encoder_hidden_states"]
                    ).squeeze()  # (bs,)
                    losses_logvar = (log_w - F_t) ** 2  # (bs,)
                    info["loss"].append(losses_logvar.mean().detach())
                    info["f_psi"].append(F_t.mean().detach())
                    losses = losses_logvar

                    if config['train']['klreg'] > 0:
                        losses = losses + config['train']['klreg'] * klreg
                        info["klreg"].append(klreg.mean().detach())
                    loss = torch.mean(losses)

                    if logr_tmp is not None:
                        info["logr"].append(torch.mean(logr_tmp).detach())
                    if logr_next_tmp is not None:
                        info["logr_next"].append(torch.mean(logr_next_tmp).detach())

                    loss = loss / accumulation_steps
                    if scaler:
                        # Backward passes under autocast are not recommended
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # prevent OOM
                    image = None
                    latent = latent_next = None
                    model_output = ref_model_output = None
                    uncond_logits = cond_logits = None
                    log_prob_diff = logr = logr_next = logr_next_tmp = logr_tmp = None
                    klreg = losses_logvar = losses = loss = None

                if ((j == num_train_timesteps - 1) and
                        (i + 1) % config['train']['gradient_accumulation_steps'] == 0):
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer.parameters(), config['train']['max_grad_norm'])
                        torch.nn.utils.clip_grad_norm_(f_psi.parameters(), config['train']['max_grad_norm'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(transformer.parameters(), config['train']['max_grad_norm'])
                        torch.nn.utils.clip_grad_norm_(f_psi.parameters(), config['train']['max_grad_norm'])
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    dist.barrier()
                    for k, v in info.items():
                        dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    info = {k: v / num_processes for k, v in info.items()}
                    for k, v in info.items():
                        result[k][global_step] = v.item()

                    info.update({"epoch": epoch})
                    result["epoch"][global_step] = epoch
                    result["time"][global_step] = time.time() - start_time

                    if is_local_main_process:
                        if config['wandb']:
                            wandb.log(info, step=global_step)
                        logger.info(f"global_step={global_step}  " +
                              " ".join([f"{k}={v:.3f}" for k, v in info.items()]))
                    info = defaultdict(list) # reset info dict

        if is_local_main_process:
            pickle.dump(result, gzip.open(os.path.join(output_dir, f"result.json"), 'wb'))
        dist.barrier()

        if epoch % config['save_freq'] == 0 or epoch == config['num_epochs'] - 1:
            if is_local_main_process:
                save_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}")
                unwrapped_unet = unwrap_model(transformer)
                transformer_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )
                Pipeline.save_lora_weights(
                    save_directory=save_path,
                    transformer_lora_layers=transformer_lora_state_dict,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )
                logger.info(f"Saved state to {save_path}")

                if config['wandb'] and config['wandb_sync_artifacts']:
                   artifact = wandb.Artifact(
                       name=f"{name}_ckpt_epoch{epoch}",
                       type="model",
                       description=f"Model checkpoint for {name} at epoch {epoch}",
                       metadata={"epoch": epoch, "global_step": global_step, "name": name},
                   )
                   artifact.add_file(f"{save_path}/pytorch_lora_weights.safetensors")
                   wandb.log_artifact(artifact)

            dist.barrier()

    if config['wandb'] and is_local_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
    dist.destroy_process_group()