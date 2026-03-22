## DNA Design (Biology)

This codebase is built on top of [DRAKES](https://github.com/ChenyuWang-Monica/DRAKES) (Wang et al.)

---

### Installation

```bash
conda create -n smc-dna python=3.9.18
conda activate smc-dna
```

Install gReLU (v1.0.2, required for the reward oracle):

```bash
git clone --branch v1.0.2 https://github.com/Genentech/gReLU.git
cd gReLU
pip install .
cd ..
```

Install remaining dependencies:

```bash
cd biology_design/
bash env.sh
```

Log in to Weights & Biases (required to download Enformer weights on first run):

```bash
wandb login
```

---

### Data and pretrained models

Download the data and model weights from the [DRAKES Dropbox link](https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0) (see their [README](https://github.com/ChenyuWang-Monica/DRAKES/blob/master/drakes_dna/README.md)) and save the contents to a local directory.
Set the environment variable before running:

```bash
export DRAKES_DATA_ROOT=/path/to/data_and_model
```

---

### Training

Run the finetuning script with the final paper hyperparameters:

```bash
cd biology_design/
bash scripts/train.sh
```

Or run directly with custom hyperparameters:

```bash
LD_LIBRARY_PATH= python train.py \
  finetuning.alpha=0.001 \
  finetuning.kl_weight=0.0 \
  finetuning.kl_method=forward \
  finetuning.entropy_weight=2.5
```

All config options can be found in `configs/train.yaml`.

---

### Evaluation

Pass the checkpoint saved during training via `smc.ft_model_ckpt`:

```bash
cd biology_design/
bash scripts/eval.sh <ft_model_ckpt>
# e.g. bash scripts/eval.sh model_weights/20260322/004927/ckpt_10/model.pth
```

All config options can be found in `configs/eval.yaml`.

Reported metrics: Pred-Activity (median predicted enhancer activity), ATAC-Acc (chromatin accessibility), 3-mer Pearson correlation, JASPAR motif Spearman correlation, and log-likelihood under the pretrained model.

