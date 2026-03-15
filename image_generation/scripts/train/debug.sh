export CUDA_VISIBLE_DEVICES="0"
n_device=1

torchrun --standalone --nproc_per_node=$n_device train.py \
    output_dir=./output/debug \
    train.klreg="0.0e+0" \
    train.reward_exp="1.0e+2" \
    train.num_rewardest_samples=1 \
    wandb_mode=disabled