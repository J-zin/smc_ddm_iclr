from typing import Optional, Tuple, Callable
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vq_model import VQModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from meissonic.smc.transformer import Transformer2DModel
from meissonic.smc.scheduler import BaseScheduler
from meissonic.smc.resampling import compute_ess_from_log_w, normalize_weights
from meissonic.lora_pipeline import MeissonicLoraLoaderMixin


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """
    Build positional IDs for latent-image tokens.

    Each latent token corresponds to a downsampled image “pixel” in a 2D grid.
    This function creates a (H//2, W//2, 3) grid where:
      - channel 0 is reserved (all zeros)
      - channel 1 stores the row index (0 .. H//2-1)
      - channel 2 stores the column index (0 .. W//2-1)

    Args:
        batch_size (int):   Number of images in the batch (unused here, but kept for API consistency).
        height (int):       Input image height (pre-VAE) or latent height depending on call site.
        width (int):        Input image width (pre-VAE) or latent width depending on call site.
        device (torch.device):  Device on which to place the returned tensor.
        dtype (torch.dtype):    Desired data type of the returned tensor.

    Returns:
        torch.Tensor of shape ((H//2 * W//2), 3) with dtype and device as specified.
          Each row is [0, row_index, col_index], flattened in row-major order.
    """
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def logmeanexp(x, dim=None, keepdim=False):
    """Numerically stable log-mean-exp using torch.logsumexp."""
    if dim is None:
        x = x.view(-1)
        dim = 0
    # log-sum-exp with or without keeping the reduced dim
    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    # subtract log(N) to convert sum into mean (broadcasts correctly)
    return lse - math.log(x.size(dim))


class Pipeline(
    DiffusionPipeline,
    MeissonicLoraLoaderMixin,
):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: Transformer2DModel 
    scheduler: BaseScheduler

    model_cpu_offload_seq = "text_encoder->transformer->vqvae"
    
    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: Transformer2DModel, 
        scheduler: BaseScheduler,
        # device,
        # model_dtype: torch.dtype = torch.float,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            # tokenizer_t5=tokenizer_t5,
            # text_encoder_t5=text_encoder_t5,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

        # self.vqvae = vqvae.to(device)
        # self.tokenizer = tokenizer
        # self.text_encoder = text_encoder.to(device)
        # self.transformer = transformer.to(device)
        # self.scheduler = scheduler
        # self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1) # type: ignore
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)
        # self._execution_device = device
        self.model_dtype = torch.bfloat16
        
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        reward_fn: Callable,
        resample_fn: Callable,
        resample_frequency: int = 1,
        kl_weight: float = 1.0,
        lambdas: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 48,
        guidance_scale: float = 9.0,
        negative_prompt = None,
        num_particles: int = 1,
        batch_p: int = 1,
        phi: int = 1, # number of samples for reward approximation
        tau: float = 1.0, # temperature for taking x0 samples
        output_type="pil",
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        proposal_type:str = "locally_optimal",
        use_continuous_formulation: bool = False, # Whether to use a continuous formulation of carry over unmasking
        disable_progress_bar: bool = False,
        verbose=True,
    ):

        # Set default lambdas
        if lambdas is None:
            lambdas = torch.ones(num_inference_steps + 1)
        assert len(lambdas) == num_inference_steps + 1, f"lambdas must of length {num_inference_steps + 1}"
        lambdas = lambdas.clamp_min(0.001).to(self._execution_device)
        
        # 1. Calculate prompt (and negative prompt) embeddings
        prompt = [prompt]
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids.to(self._execution_device)
        outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
        prompt_embeds = outputs.text_embeds
        encoder_hidden_states = outputs.hidden_states[-2]
        prompt_embeds = prompt_embeds.repeat(batch_p, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(batch_p, 1, 1)
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = [""]
            else:
                negative_prompt = [negative_prompt]
            input_ids = self.tokenizer(
                negative_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).input_ids.to(self._execution_device)
            outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
            negative_prompt_embeds = outputs.text_embeds
            negative_encoder_hidden_states = outputs.hidden_states[-2]
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_p, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(batch_p, 1, 1)
            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])
        
        # 2. Prepare micro-conditions
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_p if guidance_scale > 1.0 else batch_p, -1)
        
        # 3. Intialize latents
        shape = (num_particles, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.full(
            shape, self.scheduler.mask_token_id, dtype=torch.long, device=self._execution_device # type: ignore
        )
        vocab_size = self.transformer.config.vocab_size # type:ignore
        codebook_size = self.transformer.config.codebook_size # type: ignore
        
        # Set some constant vectors
        ONE = torch.ones(vocab_size, device=self._execution_device).float()
        MASK = F.one_hot(torch.tensor(self.scheduler.mask_token_id), num_classes=vocab_size).float().to(self._execution_device) # type: ignore
        
        # 4. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 5. Set SMC variables
        logits = torch.zeros((*latents.shape, vocab_size), device=self._execution_device)
        rewards = torch.zeros((num_particles,), device=self._execution_device)
        rewards_grad = torch.zeros((*latents.shape, vocab_size), device=self._execution_device)
        log_twist = torch.zeros((num_particles, ), device=self._execution_device)
        log_prob_proposal = torch.zeros((num_particles, ), device=self._execution_device)
        log_prob_diffusion = torch.zeros((num_particles, ), device=self._execution_device)
        log_w = torch.zeros((num_particles, ), device=self._execution_device)
        
        def propagate():
            if proposal_type == "locally_optimal":
                propgate_locally_optimal()
            elif proposal_type == "reverse":
                propagate_reverse()
            elif proposal_type == "without_SMC":
                propagate_without_SMC()
            else:
                raise NotImplementedError(f"Proposal type {proposal_type} is not implemented.")
            
        def propgate_locally_optimal():
            """SMC_grad"""

            nonlocal log_w, latents, log_prob_proposal, log_prob_diffusion, logits, rewards, rewards_grad, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                with torch.enable_grad():
                    latents_one_hot = F.one_hot(latents_batch, num_classes=vocab_size).to(dtype=self.model_dtype).requires_grad_(True)
                    if guidance_scale > 1.0:
                        # Latents are duplicated to get both unconditional and conditional logits
                        model_input = torch.cat([latents_one_hot] * 2) # type: ignore
                    else:
                        model_input = latents_one_hot
                    # img_ids, text_ids are used for positional embeddings
                    if height == 1024: #args.resolution == 1024:
                        img_ids = _prepare_latent_image_ids(model_input.shape[0], model_input.shape[1],model_input.shape[2],model_input.device,model_input.dtype)
                    else:
                        img_ids = _prepare_latent_image_ids(model_input.shape[0],2*model_input.shape[1],2*model_input.shape[2],model_input.device,model_input.dtype)
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = encoder_hidden_states.device, dtype = encoder_hidden_states.dtype)
                    model_output = self.transformer(
                        hidden_states = model_input,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        img_ids = img_ids,
                        txt_ids = txt_ids,
                        timestep = torch.tensor([timestep], device=model_input.device, dtype=torch.long),
                    )
                    if guidance_scale > 1.0:
                        uncond_logits, cond_logits = model_output.chunk(2)
                        model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                    tmp_logits = torch.permute(model_output, (0, 2, 3, 1)).float()
                    pad_logits = torch.full(
                        (*tmp_logits.shape[:3], vocab_size - codebook_size),
                        -torch.inf, 
                        device=tmp_logits.device, dtype=tmp_logits.dtype
                    )
                    tmp_logits = torch.cat([tmp_logits, pad_logits], dim=-1)
                    
                    tmp_rewards = torch.zeros(latents_batch.size(0), phi, device=self._execution_device)
                    gamma = 1 - ((ONE - MASK) * latents_one_hot).sum(dim=-1, keepdim=True)
                    for phi_i in range(phi):
                        sample = F.gumbel_softmax(tmp_logits, tau=tau, hard=True)
                        if use_continuous_formulation:
                            sample = gamma * sample + (ONE - MASK) * latents_one_hot
                        sample = self._decode_one_hot_latents(sample, batch_p, height, width, "pt")
                        tmp_rewards[:, phi_i] = reward_fn(sample)
                    tmp_rewards = logmeanexp(tmp_rewards * scale_cur, dim=-1) / scale_cur
                    
                    tmp_rewards_grad = torch.autograd.grad(
                        outputs=tmp_rewards, 
                        inputs=latents_one_hot,
                        grad_outputs=torch.ones_like(tmp_rewards),
                    )[0].detach()
                
                logits[j:j+batch_p] = tmp_logits.detach()
                rewards[j:j+batch_p] = tmp_rewards.detach()
                rewards_grad[j:j+batch_p] = tmp_rewards_grad.detach()
                log_twist[j:j+batch_p] = rewards[j:j+batch_p] * scale_cur
                
            if verbose:
                print("Rewards: ", rewards)
            
            # Calculate weights
            incremental_log_w = (log_prob_diffusion - log_prob_proposal) + (log_twist - log_twist_prev)
            log_w += incremental_log_w
            
            if verbose:
                print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
                print("log_twist - log_twist_prev:", log_twist - log_twist_prev)
                print("Incremental log weights: ", incremental_log_w)
                print("Log weights: ", log_w)
                print("Normalized weights: ", normalize_weights(log_w))
            
            # Resample particles
            if verbose:
                print(f"ESS: ", compute_ess_from_log_w(log_w))
            if resample_condition:
                resample_indices, is_resampled, log_w = resample_fn(log_w)
                if is_resampled:
                    latents = latents[resample_indices]
                    logits = logits[resample_indices]
                    rewards = rewards[resample_indices]
                    rewards_grad = rewards_grad[resample_indices]
                    log_twist = log_twist[resample_indices]
                if verbose:
                    print("Resample indices: ", resample_indices)
            
            # Propose new particles
            sched_out = self.scheduler.step_with_approx_guidance(
                latents=latents,
                step=i,
                logits=logits,
                approx_guidance=rewards_grad * scale_next
            )
            if verbose:
                print("Approx guidance norm: ", ((rewards_grad * scale_next) ** 2).sum(dim=(1, 2, 3)).sqrt())
            latents, log_prob_proposal, log_prob_diffusion = (
                sched_out.new_latents,
                sched_out.log_prob_proposal,
                sched_out.log_prob_diffusion,
            )
          
        def propagate_reverse():
            """This is a reverse SMC proposal (SMC_base or SMC_amot) that first proposes new particles and then calculates rewards and weights."""
            
            nonlocal log_w, latents, logits, rewards, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                if guidance_scale > 1.0:
                    # Latents are duplicated to get both unconditional and conditional logits
                    model_input = torch.cat([latents_batch] * 2) # type: ignore
                else:
                    model_input = latents_batch
                # img_ids, text_ids are used for positional embeddings
                if height == 1024: #args.resolution == 1024:
                    img_ids = _prepare_latent_image_ids(model_input.shape[0], model_input.shape[1],model_input.shape[2],model_input.device,model_input.dtype)
                else:
                    img_ids = _prepare_latent_image_ids(model_input.shape[0],2*model_input.shape[1],2*model_input.shape[2],model_input.device,model_input.dtype)
                txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = encoder_hidden_states.device, dtype = encoder_hidden_states.dtype)
                model_output = self.transformer(
                    hidden_states = model_input,
                    micro_conds=micro_conds,
                    pooled_projections=prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    img_ids = img_ids,
                    txt_ids = txt_ids,
                    timestep = torch.tensor([timestep], device=model_input.device, dtype=torch.long),
                )
                if guidance_scale > 1.0:
                    uncond_logits, cond_logits = model_output.chunk(2)
                    model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                tmp_logits = torch.permute(model_output, (0, 2, 3, 1)).float()
                pad_logits = torch.full(
                    (*tmp_logits.shape[:3], vocab_size - codebook_size),
                    -torch.inf, 
                    device=tmp_logits.device, dtype=tmp_logits.dtype
                )
                tmp_logits = torch.cat([tmp_logits, pad_logits], dim=-1)
                
                tmp_rewards = torch.zeros(latents_batch.size(0), phi, device=self._execution_device)
                for phi_i in range(phi):
                    sample = F.gumbel_softmax(tmp_logits, tau=tau, hard=True).argmax(dim=-1)
                    sample = self._decode_latents(sample, batch_p, height, width, "pt")
                    tmp_rewards[:, phi_i] = reward_fn(sample)
                tmp_rewards = logmeanexp(tmp_rewards * scale_cur, dim=-1) / scale_cur
                
                logits[j:j+batch_p] = tmp_logits.detach()
                rewards[j:j+batch_p] = tmp_rewards.detach()
                log_twist[j:j+batch_p] = rewards[j:j+batch_p] * scale_cur
                
            if verbose:
                print("Rewards: ", rewards)
            
            # Calculate weights
            incremental_log_w = (log_twist - log_twist_prev)
            log_w += incremental_log_w
            
            if verbose:
                print("log_twist - log_twist_prev:", log_twist - log_twist_prev)
                print("Incremental log weights: ", incremental_log_w)
                print("Log weights: ", log_w)
                print("Normalized weights: ", normalize_weights(log_w))
            
            # Resample particles
            if verbose:
                print(f"ESS: ", compute_ess_from_log_w(log_w))
            if resample_condition:
                resample_indices, is_resampled, log_w = resample_fn(log_w)
                if is_resampled:
                    latents = latents[resample_indices]
                    logits = logits[resample_indices]
                    rewards = rewards[resample_indices]
                    log_twist = log_twist[resample_indices]
                if verbose:
                    print("Resample indices: ", resample_indices)
            
            # Propose new particles
            sched_out = self.scheduler.step(
                latents=latents,
                step=i,
                logits=logits,
            )
            latents = sched_out.new_latents
            
        def propagate_without_SMC():
            """This is a simple propagation that does not use any SMC weighting or resampling (BoN)."""

            nonlocal latents, logits
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                if guidance_scale > 1.0:
                    # Latents are duplicated to get both unconditional and conditional logits
                    model_input = torch.cat([latents_batch] * 2) # type: ignore
                else:
                    model_input = latents_batch
                # img_ids, text_ids are used for positional embeddings
                if height == 1024: #args.resolution == 1024:
                    img_ids = _prepare_latent_image_ids(model_input.shape[0], model_input.shape[1],model_input.shape[2],model_input.device,model_input.dtype)
                else:
                    img_ids = _prepare_latent_image_ids(model_input.shape[0],2*model_input.shape[1],2*model_input.shape[2],model_input.device,model_input.dtype)
                txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = encoder_hidden_states.device, dtype = encoder_hidden_states.dtype)
                model_output = self.transformer(
                    hidden_states = model_input,
                    micro_conds=micro_conds,
                    pooled_projections=prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    img_ids = img_ids,
                    txt_ids = txt_ids,
                    timestep = torch.tensor([timestep], device=model_input.device, dtype=torch.long),
                )
                if guidance_scale > 1.0:
                    uncond_logits, cond_logits = model_output.chunk(2)
                    model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                tmp_logits = torch.permute(model_output, (0, 2, 3, 1)).float()
                pad_logits = torch.full(
                    (*tmp_logits.shape[:3], vocab_size - codebook_size),
                    -torch.inf, 
                    device=tmp_logits.device, dtype=tmp_logits.dtype
                )
                tmp_logits = torch.cat([tmp_logits, pad_logits], dim=-1)
                logits[j:j+batch_p] = tmp_logits.detach()
            
            # Propose new particles
            sched_out = self.scheduler.step(
                latents=latents,
                step=i,
                logits=logits,
            )
            latents = sched_out.new_latents
                
        bar = enumerate(reversed(range(num_inference_steps)))
        if not disable_progress_bar:
            bar = tqdm(bar, leave=False)
        for i, timestep in bar:
            resample_condition = (i + 1) % resample_frequency == 0
            scale_cur = lambdas[i] / kl_weight
            scale_next = lambdas[i + 1] / kl_weight
            if verbose:
                print(f"scale_cur: {scale_cur}, scale_next: {scale_next}")
            propagate()
            if verbose:
                print('\n\n')
        
        # Decode latents
        outputs = []
        for j in range(0, num_particles, batch_p):
            latents_batch = latents[j:j+batch_p]
            outputs.extend(
                self._decode_latents(latents_batch, batch_p, height, width, output_type) # type: ignore
            )
        if output_type == "pt":
            outputs = torch.stack(outputs, dim=0)
        return outputs

    def _decode_latents(self, latents, batch_size, height, width, output_type):
        if output_type == "latent":
            output = latents
        else:
            needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast # type: ignore
            if needs_upcasting:
                self.vqvae.float()
            output = self.vqvae.decode(
                latents,
                force_not_quantize=True,
                shape=(
                    batch_size,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                    self.vqvae.config.latent_channels, # type: ignore
                ),
            ).sample.clip(0, 1) # type: ignore
            output = self.image_processor.postprocess(output, output_type)
            if needs_upcasting:
                self.vqvae.half()
        return output
            
    def _decode_one_hot_latents(self, latents_one_hot, batch_size, height, width, output_type):
        shape = (
            batch_size,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
            self.vqvae.config.latent_channels, # type: ignore
        )
        codebook_size = self.transformer.config.codebook_size #type: ignore
        
        needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast # type: ignore
        if needs_upcasting:
            self.vqvae.float()
        
        # get quantized latent vectors
        embedding = self.vqvae.quantize.embedding.weight
        h: torch.Tensor = latents_one_hot[..., :codebook_size].to(embedding.dtype) @ embedding
        h = h.view(shape)
        # reshape back to match original input shape
        h = h.permute(0, 3, 1, 2).contiguous()

        # Setting lookup_from_codebook to False, as we already have the codebook embeddings in h
        self.vqvae.config.lookup_from_codebook = False # type: ignore
        output = self.vqvae.decode(
            h, # type: ignore
            force_not_quantize=True,
        ).sample.clip(0, 1) # type: ignore
        self.vqvae.config.lookup_from_codebook = True # type: ignore
        
        output = self.image_processor.postprocess(output, output_type)

        if needs_upcasting:
            self.vqvae.half()
            
        return output