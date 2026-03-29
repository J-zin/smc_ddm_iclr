import os
import sys
sys.path.append('mdlm')
import math
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from peft import LoraConfig, get_peft_model
import wandb
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig

from toxicity_classifier.scorer import ToxicityScorer
from ppl.gpt2_ppl import compute_perplexity
from mdlm_diffusion import MDLMDiffusion


def _load_from_checkpoint(config, tokenizer, device):
    """Load model from checkpoint"""
    if 'hf' in config.backbone:
        return MDLMDiffusion(config, tokenizer=tokenizer).to(device)

    return MDLMDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


def summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, Trainable params: {trainable:,}")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config: DictConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from mdlm import dataloader
    tokenizer = dataloader.get_tokenizer(config)

    toxicity_scorer = ToxicityScorer()

    @torch.no_grad()
    def compute_rewards(tokens) -> torch.Tensor:
        """
        takes integer tokens directly
        """
        texts = tokenizer.batch_decode(tokens)
        scores = toxicity_scorer.score_texts(texts)
        return scores

    @torch.no_grad()
    def compute_rewards_scaled(tokens) -> torch.Tensor:
        """
        takes integer tokens directly
        """
        return compute_rewards(tokens) / config.finetuning.alpha

    @torch.no_grad()
    def estimate_rewards_scaled(probs, num_samples, method='mean'):
        B = probs.shape[0]
        dist = torch.distributions.Categorical(probs=probs)
        samples = dist.sample((num_samples,)).reshape(num_samples * B, -1) # type: ignore
        rewards = compute_rewards_scaled(samples).reshape(num_samples, B)
        if method == 'mean':
            return rewards.mean(dim=0) # E[r(x)/alpha]
        elif method == 'logmeanexp':
            return rewards.logsumexp(dim=0) - math.log(num_samples) # log E[exp(r(x)/alpha)]
        else:
            raise ValueError(f"Unknown method: {method}")

    p_ref = _load_from_checkpoint(config, tokenizer, device)
    p_ref.eval()

    q_phi = _load_from_checkpoint(config, tokenizer, device)
    q_phi.eval()
    f_psi = torch.nn.Parameter(torch.zeros(config.finetuning.num_timesteps, device=q_phi.device))

    if config.finetuning.lora.enabled:
        # LORA stuff
        lora_config = LoraConfig(
            target_modules=list(config.finetuning.lora.target_modules),
            r=config.finetuning.lora.r,
            lora_alpha=config.finetuning.lora.lora_alpha,
            lora_dropout=config.finetuning.lora.lora_dropout,
            bias=config.finetuning.lora.bias,
        )
        q_phi.backbone = get_peft_model(q_phi.backbone, lora_config) # type: ignore

    summary(q_phi)

    trainable_params = list(filter(lambda p: p.requires_grad, q_phi.parameters())) + [f_psi]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.finetuning.lr)

    base_dir = 'model_weights'
    timestamp = datetime.now().strftime("%Y%m%d/%H%M%S")
    model_save_dir = os.path.join(base_dir, timestamp)

    # Note: wandb.init() must come after model/oracle loads if any library
    # calls wandb.login() internally (e.g. gReLU), which returns None after wandb.init().
    run = wandb.init(
        project=config.wandb.project,
        config=OmegaConf.to_container(config.finetuning, resolve=True),
    )
    run.config.model_save_dir = model_save_dir

    os.makedirs(model_save_dir, exist_ok=True)

    # Save config and metadata files
    OmegaConf.save(config=config, f=f'{model_save_dir}/config.yaml')

    loss_trace = []
    reward_trace = []

    L = q_phi.config.model.length
    eps=1e-5
    timesteps = torch.linspace(1, eps, config.finetuning.num_timesteps + 1, device=q_phi.device)
    dt = (1 - eps) / config.finetuning.num_timesteps

    # Training loop
    for epoch in range(config.finetuning.num_epochs):
        run.log({"epoch": epoch+1})
        total_epoch_loss = 0.0
        for batch_idx in range(config.finetuning.batches_per_epoch):
            q_phi.train()

            # Clear all grads
            optimizer.zero_grad()

            rewards_prev = None
            log_prob_p_ref = None
            log_prob_q_phi = None
            total_loss_for_all_timesteps = 0.0
            total_log_variance_loss_for_all_timesteps = 0.0
            total_kl_loss_for_all_timesteps = 0.0
            total_kl_div_for_all_timesteps = 0.0
            kl_loss = torch.tensor(0.0, device=q_phi.device)

            # Generate batch_size samples from q_phi
            z_t = q_phi._sample_prior(config.finetuning.batch_size, L, prompt_ids=None).to(q_phi.device) # type: ignore
            for i in range(config.finetuning.num_timesteps, 0, -1):
                t = timesteps[config.finetuning.num_timesteps - i] * torch.ones(z_t.shape[0], 1, device=q_phi.device)
                # Invoke pretrained and finetune models
                with torch.enable_grad():
                    q_phi_zs_given_zt, q_phi_z0_given_zt = q_phi._sample_step(z_t, t, dt)
                with torch.no_grad():
                    p_ref_zs_given_zt, p_ref_z0_given_zt = p_ref._sample_step(z_t, t, dt)

                # Estimate rewards
                rewards = estimate_rewards_scaled(p_ref_z0_given_zt, config.finetuning.num_samples_for_reward_estimate, method=config.finetuning.reward_estimate_method)

                if i < config.finetuning.num_timesteps:
                    # Sanity checks
                    assert rewards is not None and rewards_prev is not None
                    assert log_prob_p_ref is not None and log_prob_q_phi is not None
                    assert log_prob_q_phi.requires_grad

                    log_w = (rewards - rewards_prev) + (log_prob_p_ref - log_prob_q_phi) # Shape: (batch-size,)
                    log_variance = (log_w - f_psi[i]) ** 2
                    log_variance_loss = log_variance.mean(dim=0) # take mean across batch dimension
                    total_log_variance_loss_for_all_timesteps += log_variance_loss.item()
                    run.log({"log_variance_loss_per_timestep": log_variance_loss.item()})

                    total_loss = log_variance_loss + kl_loss
                    total_loss_for_all_timesteps += total_loss.item()
                    run.log({"total_loss_per_timestep": total_loss.item()})

                    # Accumulate gradients
                    total_loss.backward()


                if config.finetuning.kl_method == 'forward':
                    kld_batch = torch.where(
                        p_ref_z0_given_zt > 0,
                        p_ref_z0_given_zt * (torch.log(p_ref_z0_given_zt) - torch.log(q_phi_z0_given_zt.clamp_min(1e-12))),
                        torch.zeros_like(p_ref_z0_given_zt)
                    ).sum(dim=(1, 2))
                elif config.finetuning.kl_method == 'backward':
                    kld_batch = torch.where(
                        q_phi_z0_given_zt > 0,
                        q_phi_z0_given_zt * (torch.log(q_phi_z0_given_zt.clamp_min(1e-12)) - torch.log(p_ref_z0_given_zt.clamp_min(1e-12))),
                        torch.zeros_like(q_phi_z0_given_zt)
                    ).sum(dim=(1, 2))
                else:
                    raise ValueError(f"Unknown KL method: {config.finetuning.kl_method}")

                kl_loss = config.finetuning.kl_weight * kld_batch.mean(dim=0) # take mean across batch dimension
                total_kl_loss_for_all_timesteps += kl_loss.item()
                total_kl_div_for_all_timesteps += kld_batch.mean(dim=0).item()
                run.log({"kl_loss_per_timestep": kl_loss.item(), "kl_div_per_timestep": kld_batch.mean(dim=0).item()})

                q_phi_dist = torch.distributions.Categorical(probs=q_phi_zs_given_zt)
                p_ref_dist = torch.distributions.Categorical(probs=p_ref_zs_given_zt)

                if config.finetuning.sample_onpolicy:
                    z_s = q_phi_dist.sample()
                else:
                    z_s = p_ref_dist.sample()

                log_prob_q_phi = q_phi_dist.log_prob(z_s).sum(dim=1)
                log_prob_p_ref = p_ref_dist.log_prob(z_s).sum(dim=1)

                # Update for next step
                z_t = z_s
                rewards_prev = rewards

            z_0 = z_t
            if q_phi.config.sampling.noise_removal:
                with torch.no_grad():
                    t = timesteps[-1] * torch.ones(z_0.shape[0], 1, device=q_phi.device)
                    unet_conditioning = q_phi.noise(t)[0]
                    logits = q_phi.forward(z_0, unet_conditioning)
                    z_0 = logits[:, :, :-1].argmax(dim=-1)

            # Compute rewards
            rewards = compute_rewards_scaled(z_0)
            assert rewards_prev is not None and log_prob_p_ref is not None and log_prob_q_phi is not None
            log_w = (rewards - rewards_prev) + (log_prob_p_ref - log_prob_q_phi) # Shape: (batch-size,)
            log_variance = (log_w - f_psi[0]) ** 2
            log_variance_loss = log_variance.mean(dim=0) # take mean across batch dimension
            total_log_variance_loss_for_all_timesteps += log_variance_loss.item()
            run.log({"log_variance_loss_per_timestep": log_variance_loss.item()})

            total_loss = log_variance_loss + kl_loss
            total_loss_for_all_timesteps += total_loss.item()
            run.log({"total_loss_per_timestep": total_loss.item()})

            # accumulate gradients
            total_loss.backward()

            # gradients step
            optimizer.step()

            print((f"Batch {batch_idx+1}/{config.finetuning.batches_per_epoch}, "
                f"Loss: {total_loss_for_all_timesteps}, Reward (avg): {rewards.mean(dim=0).item() * config.finetuning.alpha} "
                f"KL Loss: {total_kl_loss_for_all_timesteps}"))
            run.log({
                "total_loss": total_loss_for_all_timesteps,
                "log_variance_loss": total_log_variance_loss_for_all_timesteps,
                "kl_loss": total_kl_loss_for_all_timesteps,
                "kl_div": total_kl_div_for_all_timesteps,
                "final_reward": rewards.mean(dim=0).item() * config.finetuning.alpha
            })
            total_epoch_loss += total_loss_for_all_timesteps

        q_phi.eval()
        avg_loss = total_epoch_loss / config.finetuning.batches_per_epoch
        run.log({"epoch_avg_loss": avg_loss})

        # Read prompt jsonl file
        with open(config.finetuning.validation.prompt_file, "r") as f:
            prompts = [json.loads(line)["context_string"] for line in f]

        all_tokens = []
        for prompt in prompts:
            tokens = q_phi.sample(num_steps=config.finetuning.validation.inference_steps, prompt_text=prompt)
            all_tokens.append(tokens)
        all_tokens = torch.cat(all_tokens, dim=0)
        avg_rewards = compute_rewards(all_tokens).mean().item()
        run.log({"epoch_rewards": avg_rewards})

        # perplexity
        texts = tokenizer.batch_decode(all_tokens)
        ppl, total_ppl = compute_perplexity(
            generations=[{
                "context": "",
                "generations": texts,
            }],
            device=device,
        )
        run.log({"epoch_ppl": ppl, "epoch_total_ppl": total_ppl})
        # Create a wandb Table
        table = wandb.Table(columns=["Prompt", "Generated"])
        for prompt, gen in zip([""]*len(texts), texts):
            table.add_data(prompt, gen)
        # Log the whole table for this epoch
        run.log({f"epoch_samples": table})

        print(f"Epoch {epoch+1}/{config.finetuning.num_epochs},  Loss (avg): {avg_loss}, Reward: {avg_rewards}, PPL: {ppl}/{total_ppl}")

        ckpt_path = f'{model_save_dir}/ckpt_{epoch+1}'
        if config.finetuning.lora.enabled:
            q_phi.backbone.save_pretrained(f"{ckpt_path}/lora")
        else:
            torch.save(q_phi.state_dict(), f"{ckpt_path}/model.pth")
        # Save f_psi
        torch.save(f_psi, f"{ckpt_path}/f_psi.pth")
        # Save optimizer state
        torch.save(optimizer.state_dict(), f"{ckpt_path}/optimizer.pth")

        loss_trace.append(avg_loss)
        reward_trace.append(avg_rewards)

        # If BOTH loss and reward stop imporving, then stop training
        if (
            min(loss_trace) < min(loss_trace[-config.finetuning.patience:]) and
            max(reward_trace) > max(reward_trace[-config.finetuning.patience:])
        ):
            break

    run.finish()


if __name__ == '__main__':
    main()
