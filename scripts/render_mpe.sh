#!/bin/sh

scenario="simple_spread"
env="MPE"
num_landmarks=8
num_agents=8
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 200 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 \
    --model_dir "../results/MPE/simple_spread/rmappo/models/obs_2" \
    --use_wandb --use_ReLU\
    --use_partial_obs --obs_range 2
done
