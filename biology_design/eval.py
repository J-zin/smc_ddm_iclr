import os
import logging
import hydra
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
import torch.nn.functional as F

import oracle
import diffusion_gosai_update
import dataloader_gosai
from utils import set_seed
from grelu.interpret.motifs import scan_sequences

from smc.pipeline import Pipeline
from smc.scheduler import MDLMScheduler
from smc.resampling import resample

set_seed(0, use_cuda=True)


NUM_SAMPLE_BATCHES = 10
NUM_SAMPLES_PER_BATCH = 64

def compare_kmer(kmer1, kmer2, n_sp1, n_sp2, title):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])

def eval_get_metrics(raw_samples, detoeknized_samples, old_model):
    highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs_999 = oracle.cal_highexp_kmers(return_clss=True)

    # likelihood
    model_logl = old_model.get_likelihood(raw_samples, num_steps=128, n_samples=1)
    ll = np.median(model_logl.detach().cpu().numpy(), axis=0)

    # Pred-Activity
    generated_preds = oracle.cal_gosai_pred_new(detoeknized_samples, mode='eval')[:, 0]
    pred_act = np.median(generated_preds, axis=0)

    # ATAC-Acc
    generated_preds_atac = oracle.cal_atac_pred_new(detoeknized_samples)
    atac = (generated_preds_atac[:, 1] > 0.5).sum() / len(detoeknized_samples)

    # 3-mer
    generated_kmer = oracle.count_kmers(detoeknized_samples)
    kmer = compare_kmer(
        highexp_kmers_999, generated_kmer, n_highexp_kmers_999, len(detoeknized_samples), title=r"Finetuned"
    ).statistic

    # jasper
    jaspar_motifs = 'jaspar'
    motif_count_top = scan_sequences(highexp_seqs_999, jaspar_motifs)
    motif_count_top_sum = motif_count_top['motif'].value_counts()
    motif_count = scan_sequences(detoeknized_samples, jaspar_motifs)
    motif_count_sum = motif_count['motif'].value_counts()
    motifs_summary = pd.concat([motif_count_top_sum, motif_count_sum], axis=1)
    motifs_summary.columns = ['top_data', 'finetuned']
    corr_matrix = motifs_summary.corr(method='spearman')
    jasper = corr_matrix.iloc[0, 1] # Using .iloc (row position, column position)

    return ll, pred_act, atac, kmer, jasper


def compute_rewards(tokens, reward_model) -> torch.Tensor:
    """
    takes integer tokens directly
    """
    if tokens.ndim == 2:
        onehot_tokens = F.one_hot(tokens, num_classes=4).float()
    elif tokens.ndim == 3:
        onehot_tokens = tokens[:, :, :4]
    else:
        raise ValueError("Tokens must be 2 or 3 dimensional")
    preds = reward_model(onehot_tokens.transpose(1, 2)).squeeze()
    return preds[:, 0]

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(config):
    OmegaConf.save(config.smc, "smc_config.yaml", resolve=True)

    log_path = f"{config.smc.num_particles}_particles_{config.smc.final_strategy}.log"
    gfile_stream = open(log_path, 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_PATH = os.environ['DRAKES_DATA_ROOT']
    old_path = os.path.join(BASE_PATH, 'mdlm/outputs_gosai/pretrained.ckpt')
    old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=config)
    old_model.eval()
    smc_base_model = old_model

    ft_model = None
    if config.smc.ft_model_ckpt:
        ft_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=config)
        ft_model.load_state_dict(torch.load(config.smc.ft_model_ckpt))
        ft_model.eval()
        if config.smc.proposal_type != "ft_model":
            smc_base_model = ft_model

    reward_model = oracle.get_gosai_oracle(mode='train', base_path=BASE_PATH)
    reward_model.eval()

    scheduler = MDLMScheduler(model=smc_base_model)
    pipe = Pipeline(smc_base_model, scheduler, device, model_dtype=torch.float)

    if config.smc.lambda_tempering.enabled:
        lambda_one_at = int((config.smc.lambda_tempering.one_at / 100) * config.smc.num_inference_steps)
        lambdas = torch.cat([torch.linspace(0, 1, lambda_one_at + 1), torch.ones(config.smc.num_inference_steps - lambda_one_at)])
    else:
        lambdas = None


    ll_list = []
    pred_act_list = []
    atac_list = []
    kmer_list = []
    jasper_list = []
    for itr in range(3):

        all_detoeknized_samples = []
        all_raw_samples = []
        for _ in tqdm(range(NUM_SAMPLE_BATCHES)):
            samples = pipe(
                resample_fn=lambda log_w: resample(log_w, ess_threshold=config.smc.resampling.ess_threshold, partial=config.smc.resampling.partial),
                reward_fn=lambda x: compute_rewards(x, reward_model),
                batches=NUM_SAMPLES_PER_BATCH,
                num_particles=config.smc.num_particles,
                batch_p=config.smc.batch_p,
                resample_frequency=config.smc.resampling.frequency,
                num_inference_steps=config.smc.num_inference_steps,
                proposal_type=config.smc.proposal_type,
                ft_model=ft_model,
                use_ft_model_for_expected_reward=config.smc.use_ft_model_for_expected_reward,
                use_continuous_formulation=config.smc.use_continuous_formulation,
                kl_weight=config.smc.kl_weight,
                lambdas=lambdas,
                phi=config.smc.phi,
                tau=config.smc.tau,
                final_strategy=config.smc.final_strategy,
            )
            print(samples.shape)
            all_raw_samples.append(samples)
            detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
            all_detoeknized_samples.extend(detokenized_samples)
        all_raw_samples = torch.concat(all_raw_samples)

        ll, pred_act, atac, kmer, jasper = eval_get_metrics(all_raw_samples, all_detoeknized_samples, old_model)
        ll_list.append(ll)
        pred_act_list.append(pred_act)
        atac_list.append(atac)
        kmer_list.append(kmer)
        jasper_list.append(jasper)

        logger.info(f"Itr: {itr} Likelihood: {ll:.3f}")
        logger.info(f"Itr: {itr} Pred-Activity: {pred_act:.3f}")
        logger.info(f"Itr: {itr} ATAC-Acc: {atac:.3f}")
        logger.info(f"Itr: {itr} 3-mer Pearson Corr: {kmer:.3f}")
        logger.info(f"Itr: {itr} Jasper Pearson Corr: {jasper:.3f}")

    ll_mean, ll_std = np.mean(ll_list), np.std(ll_list)
    pred_act_mean, pred_act_std = np.mean(pred_act_list), np.std(pred_act_list)
    atac_mean, atac_std = np.mean(atac_list), np.std(atac_list)
    kmer_mean, kmer_std = np.mean(kmer_list), np.std(kmer_list)
    jasper_mean, jasper_std = np.mean(jasper_list), np.std(jasper_list)

    logger.info(f"Likelihood: {ll_mean:.3f} ± {ll_std:.3f}")
    logger.info(f"Pred-Activity: {pred_act_mean:.3f} ± {pred_act_std:.3f}")
    logger.info(f"ATAC-Acc: {atac_mean:.3f} ± {atac_std:.3f}")
    logger.info(f"3-mer Pearson Corr: {kmer_mean:.3f} ± {kmer_std:.3f}")
    logger.info(f"JASPAR Spearman Corr: {jasper_mean:.3f} ± {jasper_std:.3f}")

if __name__ == '__main__':
    main()
