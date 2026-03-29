import itertools

import torch
import torch.nn.functional as F
from rich import print

from mdlm.diffusion import Diffusion


class MDLMDiffusion(Diffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, prompt_text=None):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size

        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps

        if prompt_text is not None:
            assert isinstance(prompt_text, str)
            prompt = self.tokenizer([prompt_text], return_tensors='pt', padding=False)
            prompt_ids = prompt['input_ids'][:, :-1].to(self.device) # type: ignore
        else:
            prompt_ids = None

        x = self._sample_prior(batch_size_per_gpu, self.config.model.length, prompt_ids=prompt_ids).to( # type: ignore
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm":
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
        return x

    def _sample_prior(self, *batch_dims, prompt_ids=None):
        samples = super()._sample_prior(*batch_dims)
        if prompt_ids is not None:
            samples[:, :prompt_ids.shape[1]] = prompt_ids # type: ignore
        return samples

    def restore_model_and_sample(self, num_steps, eps=1e-5, prompt_text=None):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps, prompt_text=prompt_text)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return samples

    def get_logits(self, x, t):
        sigma_t, _ = self.noise(t)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        unet_conditioning = sigma_t
        log_p_x0: torch.Tensor = self.forward(x, unet_conditioning, return_raw_logits=True)
        log_p_x0[..., self.mask_index]  = -torch.inf # type: ignore
        return log_p_x0.float()

    def _sample_step(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t) # t
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning).float()
        assert move_chance_t.ndim == log_p_x0.ndim
        p_x0 = log_p_x0.softmax(dim=-1)
        assert torch.allclose(p_x0.sum(dim=-1), torch.tensor(1.0, device=p_x0.device), atol=1e-6), f"p_x0 off by {(p_x0.sum(dim=-1) - 1.0).abs().max()}"
        q_xs = p_x0 * (move_chance_t - move_chance_s) + F.one_hot(x, num_classes=self.vocab_size) * move_chance_s
        q_xs /= move_chance_t
        assert torch.allclose(q_xs.sum(dim=-1), torch.tensor(1.0, device=q_xs.device), atol=1e-6), f"q_xs off by {(q_xs.sum(dim=-1) - 1.0).abs().max()}"
        return q_xs, p_x0

    def sample(self, num_steps=None, eps=1e-5, prompt_text=None):
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps

        if prompt_text is not None:
            assert isinstance(prompt_text, str)
            prompt = self.tokenizer([prompt_text], return_tensors='pt', padding=False)
            prompt_ids = prompt['input_ids'][:, :-1].to(self.device) # type: ignore
        else:
            prompt_ids = None

        z_t = self._sample_prior(batch_size_per_gpu, self.config.model.length, prompt_ids=prompt_ids).to( # type: ignore
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        for i in range(num_steps, 0, -1):
            t = timesteps[num_steps - i] * torch.ones(z_t.shape[0], 1, device=self.device)
            with torch.no_grad():
                z_s_given_zt = self._sample_step(z_t, t, dt)[0]
                dist = torch.distributions.Categorical(probs=z_s_given_zt)
                z_s = dist.sample()
            z_t = z_s
        return z_t
