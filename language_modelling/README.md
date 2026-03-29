# Language Modelling — SMC-DDM

**Built on:** [MDLM](https://github.com/kuleshov-group/mdlm) (git submodule). Evaluation scripts adapted from [FK-Diffusion-Steering](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering).

## Setup

```bash
# 1. Initialise the mdlm submodule
git submodule update --init

# 2. Create conda environment
conda create -n smc-ddm-lm python=3.12
conda activate smc-ddm-lm

# 3. Install dependencies
cd language_modelling/
bash env.sh

# 4. Log in to wandb
wandb login
```

> **Note:** If you get a segmentation fault, prepend `LD_LIBRARY_PATH=` to clear conflicting system CUDA libraries.

---

## Training

```bash
# From language_modelling/
bash scripts/train.sh
```

Override hyperparameters at the command line:

```bash
LD_LIBRARY_PATH= python train.py \
  finetuning.alpha=0.1 \
  finetuning.kl_weight=0.2 \
  finetuning.kl_method=backward
```

Checkpoints are saved to `outputs/<hydra_run_dir>/model_weights/<date>/<time>/ckpt_<epoch>/`.

---

## Evaluation

Each eval script runs the full pipeline: **generate** → **convert format** → **compute metrics** (PPL, CoLA, dist-n, toxicity).

| Script | Method |
|--------|--------|
| `scripts/eval_bon.sh` | Best-of-N |
| `scripts/eval_smc_base.sh` | SMC (base) |
| `scripts/eval_smc_grad.sh` | SMC (grad) |
| `scripts/eval_smc_amot.sh <ft_ckpt>` | SMC (amot) |

```bash
# From language_modelling/
bash scripts/eval_bon.sh
bash scripts/eval_smc_base.sh
bash scripts/eval_smc_grad.sh
bash scripts/eval_smc_amot.sh outputs/<hydra_run_dir>/model_weights/<date>/<time>/ckpt_<N>/lora
```

Results are written to each run's output directory under `outputs/`.

Available metrics: `ppl#gpt2-xl`, `dist-1/2/3`, `toxic`, `toxic_ext`.
