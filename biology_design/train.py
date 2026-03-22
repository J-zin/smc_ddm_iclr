#!/usr/bin/env python
# coding: utf-8

import os
import math
import torch
import torch.nn.functional as F
from datetime import datetime
import wandb
from omegaconf import OmegaConf

import oracle
import diffusion_gosai_update
import dataloader_gosai

import hydra

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):
    BASE_PATH = config.base_path

    # Define the reward function
    reward_model_ft = oracle.get_gosai_oracle(mode='train', base_path=BASE_PATH)
    reward_model_ft.eval()

    @torch.no_grad()
    def compute_rewards(tokens) -> torch.Tensor:
        """
        takes integer tokens directly
        """
        onehot_tokens = F.one_hot(tokens, num_classes=4).float()
        preds = reward_model_ft(onehot_tokens.float().transpose(1, 2)).squeeze()
        return preds[:, 0]

    def compute_rewards_scaled(tokens, anneal_factor=1.0):
        return (compute_rewards(tokens) / config.finetuning.alpha) * anneal_factor

    @torch.no_grad()
    def estimate_rewards_scaled(probs, num_samples, method='mean', anneal_factor=1.0):
        B = probs.shape[0]
        dist = torch.distributions.Categorical(probs=probs)
        samples = dist.sample((num_samples,)).reshape(num_samples * B, -1) # type: ignore
        rewards = compute_rewards_scaled(samples, anneal_factor=anneal_factor).reshape(num_samples, B)
        if method == 'mean':
            return rewards.mean(dim=0) # E[r(x)/alpha]
        elif method == 'logmeanexp':
            return rewards.logsumexp(dim=0) - math.log(num_samples) # log E[exp(r(x)/alpha)]
        else:
            raise ValueError(f"Unknown method: {method}")

    # Define method for ATAC acc
    atac_acc_model = oracle.get_atac_model(base_path=BASE_PATH)
    atac_acc_model.eval()

    @torch.no_grad()
    def cal_atac_acc_fast(tokens):
        """
        tokens: list of sequences (tokenized)
        """
        onehot_tokens = F.one_hot(tokens, num_classes=4).float()
        preds = atac_acc_model(onehot_tokens.float().transpose(1, 2)).detach().cpu().numpy()
        preds = preds.squeeze() # numpy array with shape [n_seqs, 7]
        return (preds[:,1]>0.5).sum()/len(preds)


    pretrained_ckpt = os.path.join(BASE_PATH, 'mdlm/outputs_gosai/pretrained.ckpt')
    p_ref = diffusion_gosai_update.Diffusion.load_from_checkpoint(pretrained_ckpt, config=config)
    p_ref.eval()

    q_phi = diffusion_gosai_update.Diffusion.load_from_checkpoint(pretrained_ckpt, config=config)
    if config.finetuning.benchmark_drakes:
        drakes_ckpt = os.path.join(BASE_PATH, 'mdlm/reward_bp_results_final/finetuned.ckpt')
        q_phi.load_state_dict(torch.load(drakes_ckpt))
    if config.finetuning.start_ckpt:
        q_phi.load_state_dict(torch.load(os.path.join(config.finetuning.start_ckpt, "model.pth")))
    q_phi.eval()

    f_psi = torch.nn.Parameter(torch.zeros(config.finetuning.num_timesteps, device=q_phi.device))
    if config.finetuning.start_ckpt:
        f_psi.data = torch.load(os.path.join(config.finetuning.start_ckpt, "f_psi.pth"))

    def summary(model):
        # quick print counts
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total:,}, Trainable params: {trainable:,}")

    summary(q_phi)


    trainable_params = list(filter(lambda p: p.requires_grad, q_phi.parameters())) + [f_psi]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.finetuning.lr)
    if config.finetuning.start_ckpt:
        optimizer.load_state_dict(torch.load(os.path.join(config.finetuning.start_ckpt, "optimizer.pth")))


    # Start a new wandb run to track this script.
    # Note: wandb.init() must come after oracle/model loads because gReLU calls
    # wandb.login() internally, which returns None (not True) after wandb.init().
    run = wandb.init(
        project=config.wandb.project,
        config=OmegaConf.to_container(config.finetuning, resolve=True), # type: ignore
    )

    base_dir = 'model_weights'
    timestamp = datetime.now().strftime("%Y%m%d/%H%M%S")
    model_save_dir = os.path.join(base_dir, timestamp)
    wandb.config.model_save_dir = model_save_dir

    os.makedirs(model_save_dir, exist_ok=True)

    # Save config and metadata files
    OmegaConf.save(config=config, f=f'{model_save_dir}/config.yaml')
    OmegaConf.save(config=config.finetuning, f=f'{model_save_dir}/ft_config.yaml', resolve=True)

    loss_trace = []
    reward_trace = []
    atac_trace = []

    L = q_phi.config.model.length
    eps=1e-5
    timesteps = torch.linspace(1, eps, config.finetuning.num_timesteps + 1, device=q_phi.device)
    dt = (1 - eps) / config.finetuning.num_timesteps

    kl_weight = config.finetuning.kl_weight
    kl_div_threshold = 0
    if config.finetuning.kl_weight_annealing.enabled:
        if config.finetuning.kl_weight_annealing.method == 'disable_till_kl_div':
            kl_weight = 0.0
            if config.finetuning.kl_method == 'forward':
                kl_div_threshold = config.finetuning.kl_weight_annealing.disable_till_kl_div.forward
            elif config.finetuning.kl_method == 'backward':
                kl_div_threshold = config.finetuning.kl_weight_annealing.disable_till_kl_div.backward
            else:
                raise ValueError(f"Unknown kl_method: {config.finetuning.kl_method}")
        else:
            raise ValueError(f"Unknown kl_weight_annealing.method: {config.finetuning.kl_weight_annealing.method}")

    # Training loop
    for epoch in range(config.finetuning.num_epochs):
        wandb.log({"epoch": epoch+1})
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
            total_entropy_for_all_timesteps = 0.0
            total_entropy_loss_for_all_timesteps = 0.0
            kl_loss = torch.tensor(0.0, device=q_phi.device)
            entropy_loss = torch.tensor(0.0, device=q_phi.device)

            # Generate batch_size samples from q_phi
            z_t = q_phi._sample_prior(config.finetuning.batch_size, L).to(q_phi.device) # type: ignore
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

                    total_loss = log_variance_loss + kl_loss + entropy_loss
                    total_loss_for_all_timesteps += total_loss.item()
                    run.log({"total_loss_per_timestep": total_loss.item()})

                    # Accumulate gradients
                    if config.finetuning.truncate_backprop.enabled:
                        if i < config.finetuning.truncate_backprop.num_steps:
                            total_loss.backward()
                    else:
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

                kl_loss = kl_weight * kld_batch.mean(dim=0) # take mean across batch dimension
                total_kl_loss_for_all_timesteps += kl_loss.item()
                total_kl_div_for_all_timesteps += kld_batch.mean(dim=0).item()
                run.log({"kl_loss_per_timestep": kl_loss.item(), "kl_div_per_timestep": kld_batch.mean(dim=0).item()})


                entropy_batch = - (q_phi_z0_given_zt * torch.log(q_phi_z0_given_zt.clamp_min(1e-12))).sum(dim=(1, 2))
                total_entropy_for_all_timesteps += entropy_batch.mean(dim=0).item()
                entropy_loss = - config.finetuning.entropy_weight * entropy_batch.mean(dim=0) # take mean across batch dimension
                total_entropy_loss_for_all_timesteps += entropy_loss.item()
                run.log({"entropy_per_timestep": entropy_batch.mean(dim=0).item()})

                q_phi_dist = torch.distributions.Categorical(probs=q_phi_zs_given_zt)
                p_ref_dist = torch.distributions.Categorical(probs=p_ref_zs_given_zt)

                if config.finetuning.sampling_policy.strategy == "on_policy":
                    z_s = q_phi_dist.sample()
                elif config.finetuning.sampling_policy.strategy == "off_policy":
                    z_s = p_ref_dist.sample()
                elif config.finetuning.sampling_policy.strategy == "mixed":
                    num_on_policy_samples = int(config.finetuning.sampling_policy.frac_on_policy * config.finetuning.batch_size)
                    z_s = torch.cat([
                        q_phi_dist.sample()[:num_on_policy_samples],
                        p_ref_dist.sample()[num_on_policy_samples:]
                    ])
                else:
                    raise ValueError(f"Unknown sampling strategy: {config.finetuning.sampling_policy.strategy}")

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

            total_loss = log_variance_loss + kl_loss + entropy_loss
            total_loss_for_all_timesteps += total_loss.item()
            run.log({"total_loss_per_timestep": total_loss.item()})

            # accumulate gradients
            total_loss.backward()

            # gradients step
            optimizer.step()

            atac_acc = cal_atac_acc_fast(z_0).item() * 100.0 # in percentage

            print((f"Batch {batch_idx+1}/{config.finetuning.batches_per_epoch}, "
                f"Loss: {total_loss_for_all_timesteps}, Reward (avg): {rewards.mean(dim=0).item() * config.finetuning.alpha} "
                f"KL Loss: {total_kl_loss_for_all_timesteps}"))
            run.log({
                "total_loss": total_loss_for_all_timesteps,
                "log_variance_loss": total_log_variance_loss_for_all_timesteps,
                "kl_loss": total_kl_loss_for_all_timesteps,
                "kl_div": total_kl_div_for_all_timesteps,
                "final_reward": rewards.mean(dim=0).item() * config.finetuning.alpha,
                "final_atac_acc": atac_acc,
                "entropy": total_entropy_for_all_timesteps,
                "entropy_loss": total_entropy_loss_for_all_timesteps,
            })
            total_epoch_loss += total_loss_for_all_timesteps

            if config.finetuning.kl_weight_annealing.enabled:
                if config.finetuning.kl_weight_annealing.method == 'disable_till_kl_div':
                    if total_kl_div_for_all_timesteps >= kl_div_threshold:
                        kl_weight = config.finetuning.kl_weight
                    else:
                        kl_weight = 0.0
                else:
                    raise ValueError(f"Unknown kl_weight_annealing.method: {config.finetuning.kl_weight_annealing.method}")

        q_phi.eval()
        avg_loss = total_epoch_loss / config.finetuning.batches_per_epoch
        run.log({"epoch_avg_loss": avg_loss})

        all_tokens = []
        for validation_batch_idx in range(config.finetuning.validation.batches):
            tokens = q_phi._sample(num_steps=config.finetuning.validation.inference_steps)
            all_tokens.append(tokens)
        all_tokens = torch.cat(all_tokens, dim=0)
        avg_rewards = compute_rewards(all_tokens).mean().item()
        run.log({"epoch_rewards": avg_rewards})

        # ATAC acc
        atac_acc = cal_atac_acc_fast(all_tokens).item() * 100.0 # in percentage
        run.log({"epoch_atac_acc": atac_acc})

        # Log-likelihood
        log_l_p_ref = p_ref.get_likelihood(all_tokens, num_steps=128, n_samples=1)
        run.log({"epoch_log_lik": torch.median(log_l_p_ref.reshape(-1)).item()})

        # Save 10 samples
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(all_tokens[:config.finetuning.validation.save_samples].detach().cpu().numpy())
        # Create a wandb Table
        table = wandb.Table(columns=["Sample"])
        for detokenized_sample in detokenized_samples:
            table.add_data(detokenized_sample)
        # Log the whole table for this epoch
        run.log({f"epoch_samples": table})

        print(f"Epoch {epoch+1}/{config.finetuning.num_epochs},  Loss (avg): {avg_loss}, Reward: {avg_rewards}, ATAC Accuracy: {atac_acc}")

        ckpt_path = f'{model_save_dir}/ckpt_{epoch+1}'
        os.makedirs(ckpt_path, exist_ok=True)

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
        atac_trace.append(atac_acc)

        # If ALL of loss, reward, ATAC acc stop improving, then stop training
        if (
            min(loss_trace) < min(loss_trace[-config.finetuning.patience:]) and
            max(reward_trace) > max(reward_trace[-config.finetuning.patience:]) and
            max(atac_trace) > max(atac_trace[-config.finetuning.patience:])
        ):
            break

    run.finish()

if __name__ == '__main__':
    main()
