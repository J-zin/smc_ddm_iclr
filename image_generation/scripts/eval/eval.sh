export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
n_device=8

prompt_fn=drawbench     # drawbench   simple_animals  hpd_photo_painting
reward_fn=imagereward   # imagereward aesthetic_score hpscore

# To eval SMC_amot, set smc.proposal_type=reverse, use_lora=true
proposal_type=reverse   # smc-base: reverse; smc-grad: locally_optimal; bon: without_SMC

num_particles=2         # 1, 2, 4, 8, 16
num_inference_steps=48
resample_frequency=4
batch_p=$((num_particles < 2 ? num_particles : 2))

torchrun --standalone --nproc_per_node=$n_device eval.py \
    output_dir=./output/debug \
    use_lora=false \
    train.klreg="0.0e+0" \
    train.reward_exp="1.0e+4" \
    train.num_rewardest_samples=1 \
    sample.softmax_temperature=1.0 \
    sample.guidance_scale=5.0 \
    wandb_mode=disabled \
    prompt_fn=$prompt_fn \
    reward_fn=$reward_fn \
    smc.proposal_type=$proposal_type \
    smc.num_particles=$num_particles \
    smc.batch_p=$batch_p \
    smc.num_inference_steps=$num_inference_steps \
    smc.resample_frequency=$resample_frequency