export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
n_device=8

prompt_fn=drawbench
reward_fn=imagereward

torchrun --standalone --nproc_per_node=$n_device train.py \
    output_dir=./output/drawbench \
    train.klreg="0.0e+0" \
    train.reward_exp="1.0e+2" \
    train.num_rewardest_samples=1 \
    prompt_fn=$prompt_fn \
    reward_fn=$reward_fn \
    save_freq=50 \
    num_epochs=300 