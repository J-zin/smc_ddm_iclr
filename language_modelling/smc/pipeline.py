from typing import Optional, Tuple, Callable
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig
from rich import print
from peft import PeftModel

from mdlm import dataloader
from mdlm_diffusion import MDLMDiffusion
from smc.scheduler import BaseScheduler
from smc.resampling import compute_ess_from_log_w, normalize_weights


def _load_from_checkpoint(config, tokenizer):
    """Load model from checkpoint"""
    if 'hf' in config.backbone:
        return MDLMDiffusion(config, tokenizer=tokenizer).to('cuda')

    return MDLMDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


def logmeanexp(x, dim=None, keepdim=False):
    """Numerically stable log-mean-exp using torch.logsumexp."""
    if dim is None:
        x = x.view(-1)
        dim = 0
    # log-sum-exp with or without keeping the reduced dim
    lse = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    # subtract log(N) to convert sum into mean (broadcasts correctly)
    return lse - math.log(x.size(dim))


class Pipeline:

    def __init__(self, config: DictConfig, scheduler: BaseScheduler, device, dtype=torch.float) -> None:
        self.config = config
        self.tokenizer = dataloader.get_tokenizer(config)

        # Load model
        self.model = _load_from_checkpoint(self.config, self.tokenizer) # Pre-trained model
        if self.config.ft_model.ckpt_path:
            if self.config.ft_model.lora:
                self.model.backbone = PeftModel.from_pretrained(self.model.backbone, self.config.ft_model.ckpt_path)
            else:
                self.model.load_state_dict(torch.load(self.config.ft_model.ckpt_path))
            print(f"Loaded fine-tuned model from {self.config.ft_model.ckpt_path}")
        self.model.eval()

        self._execution_device = self.model.device
        self.scheduler = scheduler
        self.model_dtype = dtype

    @torch.no_grad()
    def __call__(
        self,
        reward_fn: Callable,
        resample_fn: Callable,
        prompt_text: Optional[str] = None,
        resample_frequency: int = 1,
        kl_weight: float = 1.0,
        lambdas: Optional[torch.Tensor] = None,
        num_inference_steps: int = 48,
        num_particles: int = 1,
        batch_p: int = 1,
        phi: int = 1, # number of samples for reward approximation
        tau: float = 1.0, # temperature for taking x0 samples
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

        # 1. Tokenize prompt
        if prompt_text is not None:
            assert isinstance(prompt_text, str)
            prompt = self.tokenizer([prompt_text], return_tensors='pt', padding=False)
            prompt_ids = prompt['input_ids'][:, :-1].to(self._execution_device) # type: ignore
            # Set prompt length in scheduler if supported
            if hasattr(self.scheduler, 'set_prompt_length'):
                self.scheduler.set_prompt_length(prompt_ids.shape[1]) # type: ignore
            else:
                print("[bold red]Warning:[/bold red] Scheduler does not support setting prompt length. The prompt may be get modified in the text samples.")
        else:
            prompt_ids = None

        # 3. Intialize latents
        latents = self.model._sample_prior(num_particles, self.config.model.length, prompt_ids=prompt_ids).to( # type: ignore
            self._execution_device
        )

        # Set some constant vectors
        vocab_size = self.model.vocab_size
        assert self.scheduler.mask_token_id == self.model.mask_index # type: ignore
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
            elif proposal_type == "straight_through_gradients":
                propagate_straight_through_gradients()
            elif proposal_type == "reverse":
                propagate_reverse()
            elif proposal_type == "without_SMC":
                propagate_without_SMC()
            else:
                raise NotImplementedError(f"Proposal type {proposal_type} is not implemented.")

        def propgate_locally_optimal():
            nonlocal log_w, latents, log_prob_proposal, log_prob_diffusion, logits, rewards, rewards_grad, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                with torch.enable_grad():
                    latents_one_hot = F.one_hot(latents_batch, num_classes=vocab_size).to(dtype=self.model_dtype).requires_grad_(True)
                    tmp_logits = self.model.get_logits(latents_one_hot, t)

                    tmp_rewards = torch.zeros(latents_batch.size(0), phi, device=self._execution_device)
                    gamma = 1 - ((ONE - MASK) * latents_one_hot).sum(dim=-1, keepdim=True)
                    for phi_i in range(phi):
                        sample = F.gumbel_softmax(tmp_logits, tau=tau, hard=True)
                        if use_continuous_formulation:
                            sample = gamma * sample + (ONE - MASK) * latents_one_hot
                        tmp_rewards[:, phi_i] = reward_fn(sample)
                    tmp_rewards = logmeanexp(tmp_rewards * scale_cur, dim=-1) / scale_cur

                    tmp_rewards_grad = torch.autograd.grad(
                        outputs=tmp_rewards,
                        inputs=latents_one_hot,
                        grad_outputs=torch.ones_like(tmp_rewards)
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
                print("Approx guidance norm: ", ((rewards_grad * scale_next) ** 2).sum(dim=(1, 2)).sqrt())
            latents, log_prob_proposal, log_prob_diffusion = (
                sched_out.new_latents,
                sched_out.log_prob_proposal,
                sched_out.log_prob_diffusion,
            )

        def propagate_straight_through_gradients():
            nonlocal log_w, latents, log_prob_proposal, log_prob_diffusion, logits, rewards, rewards_grad, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                tmp_logits = self.model.get_logits(latents_batch, t)

                # take the most likely sample
                sample = tmp_logits.argmax(dim=-1)

                with torch.enable_grad():
                    sample_one_hot = F.one_hot(sample, num_classes=vocab_size).float().requires_grad_(True)
                    tmp_rewards = reward_fn(sample_one_hot)
                    tmp_rewards_grad = torch.autograd.grad(
                        outputs=tmp_rewards,
                        inputs=sample_one_hot,
                        grad_outputs=torch.ones_like(tmp_rewards)
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
                print("Approx guidance norm: ", ((rewards_grad * scale_next) ** 2).sum(dim=(1, 2)).sqrt())
            latents, log_prob_proposal, log_prob_diffusion = (
                sched_out.new_latents,
                sched_out.log_prob_proposal,
                sched_out.log_prob_diffusion,
            )

        def propagate_reverse():
            nonlocal log_w, latents, logits, rewards, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                tmp_logits = self.model.get_logits(latents_batch, t)

                tmp_rewards = torch.zeros(latents_batch.size(0), phi, device=self._execution_device)
                for phi_i in range(phi):
                    sample = F.gumbel_softmax(tmp_logits, tau=tau, hard=True).argmax(dim=-1)
                    text_sample = self.decode_latents(sample)
                    tmp_rewards[:, phi_i] = reward_fn(text_sample)
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
            nonlocal latents, logits
            for j in range(0, num_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                tmp_logits = self.model.get_logits(latents_batch, t)
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
            t = ((timestep + 1) / num_inference_steps) * torch.ones(batch_p, 1, device=self._execution_device)
            resample_condition = (i + 1) % resample_frequency == 0
            scale_cur = lambdas[i] / kl_weight
            scale_next = lambdas[i + 1] / kl_weight
            if verbose:
                print(f"scale_cur: {scale_cur}, scale_next: {scale_next}")
            propagate()
            print('\n\n')

        # Decode latents
        text = self.decode_latents(latents)
        return latents, text

    def decode_latents(self, latents):
        return self.model.tokenizer.batch_decode(latents)
