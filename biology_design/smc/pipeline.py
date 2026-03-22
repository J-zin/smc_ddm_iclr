from typing import Optional, Tuple, Callable
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
import diffusion_gosai_update
from smc.scheduler import BaseScheduler
from smc.resampling import compute_ess_from_log_w, normalize_weights


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
    model: diffusion_gosai_update.Diffusion
    
    def __init__(
        self,
        model: diffusion_gosai_update.Diffusion,
        scheduler: BaseScheduler,
        device = torch.device('cuda'),
        model_dtype: torch.dtype = torch.float,
    ):
        self.model = model
        self.scheduler = scheduler
        self._execution_device = device
        self.model_dtype = model_dtype
        
    @torch.no_grad()
    def __call__(
        self,
        reward_fn: Callable,
        resample_fn: Callable,
        resample_frequency: int = 1,
        kl_weight: float = 1.0,
        lambdas: Optional[torch.Tensor] = None,
        num_inference_steps: int = 48,
        batches: int = 1, # Number of independent SMCs
        num_particles: int = 1, # Number of particles per SMC
        batch_p: int = 1, # Number of parallel particles
        phi: int = 1, # number of samples for reward approximation
        tau: float = 1.0, # temperature for taking x0 samples
        proposal_type:str = "locally_optimal",
        ft_model: Optional[diffusion_gosai_update.Diffusion] = None, # needs to supplied if proposal_type is ft_model
        use_ft_model_for_expected_reward: bool = False, # Whether to use the forward model for expected reward
        use_continuous_formulation: bool = False, # Whether to use a continuous formulation of carry over unmasking
        disable_progress_bar: bool = False,
        final_strategy="argmax_rewards",
        verbose=True,
    ):
        # Set default lambdas
        if lambdas is None:
            lambdas = torch.ones(num_inference_steps + 1)
        assert len(lambdas) == num_inference_steps + 1, f"lambdas must of length {num_inference_steps + 1}"
        lambdas = lambdas.clamp_min(0.001).to(self._execution_device)
        
        # 2. Prepare micro-conditions
        total_particles = batches * num_particles
        batch_p = min(batch_p, total_particles)
        L = self.model.config.model.length
        vocab_size = self.model.vocab_size
        
        # 3. Intialize latents
        latents = self.model._sample_prior(total_particles, L).to(self._execution_device)
        
        # Set some constant vectors
        ONE = torch.ones(vocab_size, device=self._execution_device).float()
        MASK = F.one_hot(torch.tensor(self.model.mask_index), num_classes=vocab_size).float().to(self._execution_device) # type: ignore
        
        # 5. Set SMC variables
        logits = torch.zeros((*latents.shape, vocab_size), device=self._execution_device)
        logits_ft_model = torch.zeros((*latents.shape, vocab_size), device=self._execution_device)
        rewards = torch.zeros((total_particles,), device=self._execution_device)
        rewards_grad = torch.zeros((*latents.shape, vocab_size), device=self._execution_device)
        log_twist = torch.zeros((total_particles, ), device=self._execution_device)
        log_prob_proposal = torch.zeros((total_particles, ), device=self._execution_device)
        log_prob_diffusion = torch.zeros((total_particles, ), device=self._execution_device)
        log_w = torch.zeros((total_particles, ), device=self._execution_device)
        
        def propagate():
            if proposal_type == "locally_optimal":
                propgate_locally_optimal()
            # elif proposal_type == "straight_through_gradients":
            #     propagate_straight_through_gradients()
            elif proposal_type == "reverse":
                propagate_reverse()
            elif proposal_type == "without_SMC":
                propagate_without_SMC()
            elif proposal_type == "ft_model":
                propagate_ft_model()
            else:
                raise NotImplementedError(f"Proposal type {proposal_type} is not implemented.")
            
        def propgate_locally_optimal():
            nonlocal log_w, latents, log_prob_proposal, log_prob_diffusion, logits, rewards, rewards_grad, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, total_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                with torch.enable_grad():
                    latents_one_hot = F.one_hot(latents_batch, num_classes=vocab_size).to(dtype=self.model_dtype).requires_grad_(True)
                    tmp_logits = self.model.get_logits(latents_one_hot, t[j:j+batch_p])
                    
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
            
            # Now reshape log_w to (batches, num_particles)
            log_w = log_w.reshape(batches, num_particles)
            
            if verbose:
                print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
                print("log_twist - log_twist_prev:", log_twist - log_twist_prev)
                print("Incremental log weights: ", incremental_log_w)
                print("Log weights: ", log_w)
                print("Normalized weights: ", normalize_weights(log_w, dim=-1))
            
            # Resample particles
            if verbose:
                print(f"ESS: ", compute_ess_from_log_w(log_w, dim=-1))
            
            if resample_condition:
                resample_indices = []
                log_w_new = []
                is_resampled = False
                for batch in range(batches):
                    resample_indices_batch, is_resampled_batch, log_w_batch = resample_fn(log_w[batch])
                    resample_indices.append(resample_indices_batch + batch * num_particles)
                    log_w_new.append(log_w_batch)
                    is_resampled = is_resampled or is_resampled_batch
                    
                resample_indices = torch.cat(resample_indices, dim=0)
                log_w = torch.cat(log_w_new, dim=0)
                    
                if is_resampled:
                    latents = latents[resample_indices]
                    logits = logits[resample_indices]
                    rewards = rewards[resample_indices]
                    rewards_grad = rewards_grad[resample_indices]
                    log_twist = log_twist[resample_indices]
                    
                if verbose:
                    print("Resample indices: ", resample_indices)
                
            if log_w.ndim == 2:
                log_w = log_w.reshape(total_particles)

            
            # Propose new particles
            sched_out = self.scheduler.step_with_approx_guidance(
                latents=latents,
                logits=logits,
                approx_guidance=rewards_grad * scale_next,
                t=t,
                next_t=t-dt,
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
            for j in range(0, total_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                with torch.no_grad():
                    tmp_logits = self.model.get_logits(latents_batch, t[j:j+batch_p])
                    
                    tmp_rewards = torch.zeros(latents_batch.size(0), phi, device=self._execution_device)
                    tmp_logp_x0 = self.model._subs_parameterization(tmp_logits, latents_batch)
                    for phi_i in range(phi):
                        sample = F.gumbel_softmax(tmp_logp_x0, tau=tau, hard=True)
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
            
            # Now reshape log_w to (batches, num_particles)
            log_w = log_w.reshape(batches, num_particles)
            
            if verbose:
                print("log_twist - log_twist_prev:", log_twist - log_twist_prev)
                print("Incremental log weights: ", incremental_log_w)
                print("Log weights: ", log_w)
                print("Normalized weights: ", normalize_weights(log_w, dim=-1))
            
            # Resample particles
            if verbose:
                print(f"ESS: ", compute_ess_from_log_w(log_w, dim=-1))
            
            if resample_condition:
                resample_indices = []
                log_w_new = []
                is_resampled = False
                for batch in range(batches):
                    resample_indices_batch, is_resampled_batch, log_w_batch = resample_fn(log_w[batch])
                    resample_indices.append(resample_indices_batch + batch * num_particles)
                    log_w_new.append(log_w_batch)
                    is_resampled = is_resampled or is_resampled_batch
                    
                resample_indices = torch.cat(resample_indices, dim=0)
                log_w = torch.cat(log_w_new, dim=0)
                    
                if is_resampled:
                    latents = latents[resample_indices]
                    logits = logits[resample_indices]
                    rewards = rewards[resample_indices]
                    log_twist = log_twist[resample_indices]
                    
                if verbose:
                    print("Resample indices: ", resample_indices)
                
            if log_w.ndim == 2:
                log_w = log_w.reshape(total_particles)

            
            # Propose new particles
            sched_out = self.scheduler.step(
                latents=latents,
                logits=logits,
                t=t,
                next_t=t-dt,
            )
            latents = sched_out.new_latents
        
        def propagate_without_SMC():
            nonlocal latents, logits
            for j in range(0, total_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                with torch.no_grad():
                    tmp_logits = self.model.get_logits(latents_batch, t[j:j+batch_p])    
                logits[j:j+batch_p] = tmp_logits.detach()
            
            # Propose new particles
            sched_out = self.scheduler.step(
                latents=latents,
                logits=logits,
                t=t,
                next_t=t-dt,
            )
            latents = sched_out.new_latents
            
        def propagate_ft_model():
            assert ft_model is not None, f"ft_model must be provided for proposal_type={proposal_type}."
            nonlocal log_w, latents, log_prob_proposal, log_prob_diffusion, logits, logits_ft_model, rewards, log_twist
            log_twist_prev = log_twist.clone()
            for j in range(0, total_particles, batch_p):
                latents_batch = latents[j:j+batch_p]
                with torch.no_grad():
                    tmp_logits = self.model.get_logits(latents_batch, t[j:j+batch_p])
                    tmp_logits_ft_model = ft_model.get_logits(latents_batch, t[j:j+batch_p])
                    
                    tmp_rewards = torch.zeros(latents_batch.size(0), phi, device=self._execution_device)
                    if use_ft_model_for_expected_reward:
                        tmp_logp_x0 = ft_model._subs_parameterization(tmp_logits_ft_model, latents_batch)
                    else:
                        tmp_logp_x0 = self.model._subs_parameterization(tmp_logits, latents_batch)
                    for phi_i in range(phi):
                        sample = F.gumbel_softmax(tmp_logp_x0, tau=tau, hard=True)
                        tmp_rewards[:, phi_i] = reward_fn(sample)
                    tmp_rewards = logmeanexp(tmp_rewards * scale_cur, dim=-1) / scale_cur
                
                logits[j:j+batch_p] = tmp_logits.detach()
                logits_ft_model[j:j+batch_p] = tmp_logits_ft_model.detach()
                rewards[j:j+batch_p] = tmp_rewards.detach()
                log_twist[j:j+batch_p] = rewards[j:j+batch_p] * scale_cur
                
            if verbose:
                print("Rewards: ", rewards)
            
            # Calculate weights
            incremental_log_w = (log_prob_diffusion - log_prob_proposal) + (log_twist - log_twist_prev)
            log_w += incremental_log_w
            
            # Now reshape log_w to (batches, num_particles)
            log_w = log_w.reshape(batches, num_particles)
            
            if verbose:
                print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
                print("log_twist - log_twist_prev:", log_twist - log_twist_prev)
                print("Incremental log weights: ", incremental_log_w)
                print("Log weights: ", log_w)
                print("Normalized weights: ", normalize_weights(log_w, dim=-1))
            
            # Resample particles
            if verbose:
                print(f"ESS: ", compute_ess_from_log_w(log_w, dim=-1))
            
            if resample_condition:
                resample_indices = []
                log_w_new = []
                is_resampled = False
                for batch in range(batches):
                    resample_indices_batch, is_resampled_batch, log_w_batch = resample_fn(log_w[batch])
                    resample_indices.append(resample_indices_batch + batch * num_particles)
                    log_w_new.append(log_w_batch)
                    is_resampled = is_resampled or is_resampled_batch
                    
                resample_indices = torch.cat(resample_indices, dim=0)
                log_w = torch.cat(log_w_new, dim=0)
                    
                if is_resampled:
                    latents = latents[resample_indices]
                    logits = logits[resample_indices]
                    logits_ft_model = logits_ft_model[resample_indices]
                    rewards = rewards[resample_indices]
                    log_twist = log_twist[resample_indices]
                    
                if verbose:
                    print("Resample indices: ", resample_indices)
                
            if log_w.ndim == 2:
                log_w = log_w.reshape(total_particles)

            
            # Propose new particles
            approx_guidance = logits_ft_model - logits # this effectively makes logits_ft_model the proposal distribution
            approx_guidance[..., self.model.mask_index] = 0.0 # avoid nan due to (inf - inf)
            sched_out = self.scheduler.step_with_approx_guidance(
                latents=latents,
                logits=logits,
                approx_guidance=approx_guidance, 
                t=t,
                next_t=t-dt,
            )
            latents, log_prob_proposal, log_prob_diffusion = (
                sched_out.new_latents,
                sched_out.log_prob_proposal,
                sched_out.log_prob_diffusion,
            )
        
        bar = enumerate(reversed(range(num_inference_steps)))
        if not disable_progress_bar:
            bar = tqdm(bar, leave=False)
        eps=1e-5
        timesteps = torch.linspace(1, eps, num_inference_steps + 1, device=self._execution_device)
        dt = (1 - eps) / num_inference_steps
        for i, timestep in bar:
            t = timesteps[i] * torch.ones(total_particles, 1, device=self._execution_device)
            resample_condition = (i + 1) % resample_frequency == 0
            scale_cur = lambdas[i] / kl_weight
            scale_next = lambdas[i + 1] / kl_weight
            if verbose:
                print(f"scale_cur: {scale_cur}, scale_next: {scale_next}")
            propagate()
            print('\n\n')
        
        
        if self.model.config.sampling.noise_removal:
            with torch.no_grad():
                for j in range(0, total_particles, batch_p):
                    latents_batch = latents[j:j+batch_p]
                    t = timesteps[-1] * torch.ones(batch_p, 1, device=self._execution_device)
                    unet_conditioning = self.model.noise(t)[0]
                    tmp_logits = self.model.forward(latents_batch, unet_conditioning)
                    logits[j:j+batch_p] = tmp_logits
                latents = logits[:, :, :-1].argmax(dim=-1)
        
        # Final SMC weights
        scale_cur = lambdas[-1] / kl_weight
        log_twist_prev = log_twist.clone()
        for j in range(0, total_particles, batch_p):
            latents_batch = latents[j:j+batch_p]
            with torch.no_grad():
                tmp_rewards = reward_fn(latents_batch)
                rewards[j:j+batch_p] = tmp_rewards
                log_twist[j:j+batch_p] = tmp_rewards * scale_cur
                
        if verbose:
            print("Rewards: ", rewards)

        # Calculate weights
        incremental_log_w = (log_prob_diffusion - log_prob_proposal) + (log_twist - log_twist_prev)
        log_w += incremental_log_w

        # Now reshape everything to (batches, num_particles) for final strategy
        log_w = log_w.reshape(batches, num_particles)
        latents = latents.reshape(batches, num_particles, L)
        rewards = rewards.reshape(batches, num_particles)
        
        if verbose:
            print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
            print("log_twist - log_twist_prev:", log_twist - log_twist_prev)
            print("Incremental log weights: ", incremental_log_w)
            print("Log weights: ", log_w)
            print("Normalized weights: ", normalize_weights(log_w, dim=-1))
        
        if final_strategy == "multinomial":
            final_indices = torch.multinomial(normalize_weights(log_w, dim=-1), num_samples=1).squeeze(-1)
        elif final_strategy == "argmax_rewards":
            final_indices = rewards.argmax(dim=-1)
        elif final_strategy == "argmax_weights":
            final_indices = log_w.argmax(dim=-1)
        else:
            raise NotImplementedError(f"Final strategy {final_strategy} is not implemented.")
        
        if verbose:
            print("Final selected indices: ", final_indices)
        
        latents = latents[
            torch.arange(batches, device=latents.device),
            final_indices
        ]
        return latents
