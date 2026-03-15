from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import math
import numpy as np
import torch
import torch.nn.functional as F

from meissonic.scheduler import mask_by_random_topk


@dataclass
class SchedulerStepOutput:
    new_latents: torch.Tensor


@dataclass
class SchedulerApproxGuidanceOutput:
    new_latents: torch.Tensor
    log_prob_proposal: torch.Tensor
    log_prob_diffusion: torch.Tensor


class BaseScheduler(ABC):
    @abstractmethod
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
    ) -> SchedulerStepOutput:
        pass
    
    @abstractmethod
    def set_timesteps(self, num_inference_steps: int):
        pass

    @abstractmethod
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
    ) -> SchedulerApproxGuidanceOutput:
        pass


def sum_masked_logits(
    logits: torch.Tensor,
    preds: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Sum logits at `preds` indices, masked by `mask`, handling invalid `preds`.

    Args:
        logits: Tensor of shape (B, H, W, C) - logits over C classes.
        preds: Tensor of shape (B, H, W) - predicted class indices.
        mask: Tensor of shape (B, H, W) - binary mask to include positions.

    Returns:
        Tensor of shape (B,) - sum of selected logits per batch item.
    """
    B, H, W, C = logits.shape
    # Ensure preds are in valid index range [0, C-1]
    valid = (preds >= 0) & (preds <= preds[mask].max())
    # Replace invalid preds with a dummy index (0), which we will mask later
    safe_preds = preds.masked_fill(~valid, 0)
    # Gather logits at predicted indices
    selected = torch.gather(logits, dim=3, index=safe_preds.unsqueeze(-1)).squeeze(-1)
    # Zero out contributions from invalid preds and masked positions
    selected = selected * valid * mask
    # Sum over H, W dimension
    return selected.sum(dim=(1, 2))

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of log(1 - exp(x)) for x < 0.
    """
    return torch.where(
        x > -1,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


class MeissonicScheduler(BaseScheduler):
    def __init__(self, 
            mask_token_id: int, 
            masking_schedule: str = "cosine",
        ):
        self.mask_token_id = mask_token_id
        self.masking_schedule = masking_schedule
    
    def set_timesteps(self, num_inference_steps: int, temperature: Union[int, Tuple[int, int], List[int]] = (2, 0), device='cuda'):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)
        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
        else:
            self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)
    
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        verbose: bool = False,
    ) -> SchedulerStepOutput:
        batch_size, height, width, vocab_size = logits.shape
        sample = latents.reshape(batch_size, height * width)
        model_output = logits.reshape(batch_size, height * width, vocab_size)

        unknown_map = sample == self.mask_token_id

        probs = model_output.softmax(dim=-1)

        device = probs.device
        probs_ = probs
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
        probs_ = probs_.reshape(-1, probs.size(-1))
        pred_original_sample = torch.multinomial(probs_, 1).to(device=device)
        pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
        pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

        timestep = self.num_inference_steps - 1 - step
        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.masking_schedule}")

            mask_len = (seq_len * mask_ratio).floor()
            # do not mask more than amount previously masked
            mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            # mask at least one
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)

            selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

            masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx].item())

            # Masks tokens with lower confidence.
            prev_sample = torch.where(masking, self.mask_token_id, pred_original_sample)

        if verbose:
            print("Unmasked:", (prev_sample != self.mask_token_id).sum(dim=1))
        prev_sample = prev_sample.reshape(batch_size, height, width)
        pred_original_sample = pred_original_sample.reshape(batch_size, height, width)
        
        return SchedulerStepOutput(new_latents=prev_sample)
        
    
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
        verbose: bool = False,
    ) -> SchedulerApproxGuidanceOutput:
        proposal_logits = logits + approx_guidance
        sched_out = self.step(latents, step, proposal_logits)
        new_latents = sched_out.new_latents
        
        newly_filled_positions = (latents != new_latents)
        if verbose:
            print("Newly filled positions:", newly_filled_positions.sum(dim=(1, 2)))
        
        log_prob_proposal = sum_masked_logits(
            logits=proposal_logits.log_softmax(dim=-1),
            preds=new_latents,
            mask=newly_filled_positions,
        )
        log_prob_diffusion = sum_masked_logits(
            logits=logits.log_softmax(dim=-1),
            preds=new_latents,
            mask=newly_filled_positions,
        )
        if verbose:
            print("log prob proposal:", log_prob_proposal)
            print("log prob diffusion:", log_prob_diffusion)
        return SchedulerApproxGuidanceOutput(
            new_latents,
            log_prob_proposal,
            log_prob_diffusion,
        )


class ReMDMScheduler(BaseScheduler):
    def __init__(
        self,
        schedule,
        remask_strategy,
        eta,
        mask_token_id,
        temperature=1.0,
    ):
        self.schedule = schedule
        self.remask_strategy = remask_strategy
        self.eta = eta 
        self.temperature = temperature
        self.mask_token_id = mask_token_id
    
    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        if self.schedule == "linear":
            self.alphas = 1 - torch.linspace(0, 1, num_inference_steps + 1)
        elif self.schedule == "cosine":
            self.alphas = 1 - torch.cos((math.pi/2) * (1 - torch.linspace(0, 1, num_inference_steps + 1)))
        else:
            raise ValueError(f"unknown masking schedule {self.schedule}")
    
    def step(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
    ) -> SchedulerStepOutput:
        B, H, W, C = logits.shape
        assert latents.shape == (B, H, W)
        
        latents = latents.reshape(B, H*W)
        logits = logits.reshape(B, H*W, C)
        
        t = self.num_inference_steps - step
        s = t - 1
        
        alpha_t = self.alphas[t]
        alpha_s = self.alphas[s]
        sigma_t_max = torch.clamp_max((1 - alpha_s) / alpha_t, 1.0)
        if self.remask_strategy == "max_cap":
            sigma_t = torch.clamp_max(sigma_t_max, self.eta)
        elif self.remask_strategy == "rescale":
            sigma_t = sigma_t_max * self.eta
        else:
            raise ValueError(f"unknown masking schedule {self.remask_strategy}")
        
        # z_t != m
        x_theta = F.one_hot(latents, num_classes=C).float()
        logits_z_t_neq_m = (
            torch.log(x_theta) +
            torch.log(1 - sigma_t)
        )
        logits_z_t_neq_m[..., self.mask_token_id] = (
            torch.log(sigma_t)
        )
        
        # z_t = m
        log_x_theta = (logits / self.temperature).log_softmax(dim=-1)
        logits_z_t_eq_m = (
            log_x_theta + 
            torch.log((alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t))
        )
        logits_z_t_eq_m[..., self.mask_token_id] = (
            torch.log((1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t))
        )
        
        z_t_neq_m = (latents != self.mask_token_id)
        p_theta_logits = torch.where(
            z_t_neq_m.unsqueeze(-1).expand(-1, -1, C),
            logits_z_t_neq_m,
            logits_z_t_eq_m,
        )
        assert torch.allclose(torch.exp(p_theta_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device)), (torch.exp(p_theta_logits).sum(dim=-1) - torch.ones(B, H*W, device=logits.device)).abs().max()
        diffusion_dist = torch.distributions.Categorical(logits=p_theta_logits) # type: ignore
        new_latents = diffusion_dist.sample()
        print("Unmasked:", (new_latents != self.mask_token_id).sum(dim=1))
        return SchedulerStepOutput(new_latents.reshape(B, H, W))
    
    def step_with_approx_guidance(
        self,
        latents: torch.Tensor,
        step: int,
        logits: torch.Tensor,
        approx_guidance: torch.Tensor,
    ) -> SchedulerApproxGuidanceOutput:
        B, H, W, C = logits.shape
        assert latents.shape == (B, H, W)
        assert approx_guidance.shape == (B, H, W, C)
        
        latents = latents.reshape(B, H*W)
        logits = logits.reshape(B, H*W, C)
        approx_guidance = approx_guidance.reshape(B, H*W, C)
        
        t = self.num_inference_steps - step
        s = t - 1
        
        alpha_t = self.alphas[t]
        alpha_s = self.alphas[s]
        sigma_t_max = torch.clamp_max((1 - alpha_s) / alpha_t, 1.0)
        if self.remask_strategy == "max_cap":
            sigma_t = torch.clamp_max(sigma_t_max, self.eta)
        elif self.remask_strategy == "rescale":
            sigma_t = sigma_t_max * self.eta
        else:
            raise ValueError(f"unknown masking schedule {self.remask_strategy}")
        
        # z_t != m
        x_theta = F.one_hot(latents, num_classes=C).float()
        logits_z_t_neq_m = (
            torch.log(x_theta) +
            torch.log(1 - sigma_t)
        )
        logits_z_t_neq_m[..., self.mask_token_id] = (
            torch.log(sigma_t)
        )
        
        # z_t = m
        log_x_theta = (logits / self.temperature).log_softmax(dim=-1)
        logits_z_t_eq_m = (
            log_x_theta + 
            torch.log((alpha_s - (1 - sigma_t) * alpha_t) / (1 - alpha_t))
        )
        logits_z_t_eq_m[..., self.mask_token_id] = (
            torch.log((1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t))
        )
        
        z_t_neq_m = (latents != self.mask_token_id)
        p_theta_logits = torch.where(
            z_t_neq_m.unsqueeze(-1).expand(-1, -1, C),
            logits_z_t_neq_m,
            logits_z_t_eq_m,
        )
        assert torch.allclose(torch.exp(p_theta_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device))
        
        proposal_logits = (p_theta_logits + approx_guidance).log_softmax(dim=-1)
        assert torch.allclose(torch.exp(proposal_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device))
        
        # modify proposal logits to have the same mask schedule as the original logits
        proposal_logits[..., :self.mask_token_id] += (
            torch.logsumexp(p_theta_logits[..., :self.mask_token_id], dim=(1, 2), keepdim=True) - 
            torch.logsumexp(proposal_logits[..., :self.mask_token_id], dim=(1, 2), keepdim=True)
        )
        proposal_logits[..., :self.mask_token_id] = torch.where(
            proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1, keepdim=True) >= 0,
            proposal_logits[..., :self.mask_token_id].log_softmax(dim=-1),
            proposal_logits[..., :self.mask_token_id]
        )
        assert not (proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1) > 1e-6).any(), proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1).max()
        proposal_logits[..., self.mask_token_id] = (
            log1mexp(proposal_logits[..., :self.mask_token_id].logsumexp(dim=-1).clamp_max(0))
        )
        assert torch.allclose(torch.exp(proposal_logits).sum(dim=-1), torch.ones(B, H*W, device=logits.device)), (torch.exp(proposal_logits).sum(dim=-1) - torch.ones(B, H*W, device=logits.device)).abs().max()
        # modify proposal logits to have the same mask schedule as the original logits
        
        proposal_dist = torch.distributions.Categorical(logits=proposal_logits) # type: ignore
        diffusion_dist = torch.distributions.Categorical(logits=p_theta_logits) # type: ignore
        
        new_latents = proposal_dist.sample()
        
        log_prob_proposal = proposal_dist.log_prob(new_latents).sum(dim=1)
        log_prob_diffusion = diffusion_dist.log_prob(new_latents).sum(dim=1)
        
        print("Unmasked:", (new_latents != self.mask_token_id).sum(dim=1))
        return SchedulerApproxGuidanceOutput(
            new_latents.reshape(B, H, W),
            log_prob_proposal,
            log_prob_diffusion,
        )