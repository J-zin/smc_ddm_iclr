from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def log_prob_diff_step_old(
    self, model_output, ref_model_output, latents, latents_next
):
    """
    Computes log_prob_p_ref - log_prob_q_phi
    """
    mask_token_id = self.scheduler.config.mask_token_id
    mask = latents == mask_token_id
    mask_next = latents_next == mask_token_id
    unmask = mask ^ mask_next
    unmask_index = unmask * latents_next

    q_psi_logprob = model_output.log_softmax(dim=1)
    p_ref_logprob = ref_model_output.log_softmax(dim=1)
    q_psi_logprob = q_psi_logprob.gather(1, unmask_index.unsqueeze(1)).squeeze(1)
    p_ref_logprob = p_ref_logprob.gather(1, unmask_index.unsqueeze(1)).squeeze(1)

    log_prob_diff = torch.where(
        unmask,
        p_ref_logprob - q_psi_logprob,
        torch.zeros_like(p_ref_logprob),
    )
    p_ref_logprob = torch.where(
        unmask,
        p_ref_logprob,
        torch.zeros_like(p_ref_logprob),
    )
    q_psi_logprob = torch.where(
        unmask,
        q_psi_logprob,
        torch.zeros_like(q_psi_logprob),
    )

    return log_prob_diff.sum(dim=(1,2)), q_psi_logprob.sum(dim=(1,2)), p_ref_logprob.sum(dim=(1,2)) # (bs,)

def log_prob_diff_step(
    self, log_q_psi, log_p_ref, latents, latents_next
):
    """
    Computes (log_p_ref - log_q_psi) on new unmasked positions
    """
    mask_token_id = self.scheduler.config.mask_token_id
    mask = latents == mask_token_id
    mask_next = latents_next == mask_token_id
    unmask = mask ^ mask_next

    log_q_psi = - F.cross_entropy(
        log_q_psi,
        latents_next * unmask,  # * unmask to avoid out of index error
        reduction="none",
    ) * unmask

    log_p_ref = - F.cross_entropy(
        log_p_ref,
        latents_next * unmask,  # * unmask to avoid out of index error
        reduction="none",
    ) * unmask

    log_prob_diff = log_p_ref - log_q_psi

    return log_prob_diff.sum(dim=(1,2)) # (bs,)

def pred_orig_latent(
    self,
    model_output: torch.Tensor,
    sample: torch.LongTensor,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
):
    two_dim_input = sample.ndim == 3 and model_output.ndim == 4

    if two_dim_input:
        batch_size, codebook_size, height, width = model_output.shape
        sample = sample.reshape(batch_size, height * width)
        model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)

    unknown_map = sample == self.config.mask_token_id

    probs = model_output.softmax(dim=-1)

    device = probs.device
    probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
    if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
        probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
    probs_ = probs_.reshape(-1, probs.size(-1))
    
    pred_original_sample = torch.multinomial(probs_, num_samples, generator=generator).to(device=device)
    pred_original_sample = pred_original_sample.view(*probs.shape[:-1], num_samples)
    pred_original_sample = torch.where(
        unknown_map.unsqueeze(-1), 
        pred_original_sample, 
        sample.unsqueeze(-1)
    )
    pred_original_sample = pred_original_sample.transpose(1, 2)  # (bs, n_samples, height*width)

    if two_dim_input:
        pred_original_sample = pred_original_sample.reshape(batch_size, num_samples, height, width)

    return pred_original_sample