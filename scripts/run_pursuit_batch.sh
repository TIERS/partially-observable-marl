#!/bin/sh

#SBATCH --job-name=pursuit
#SBATCH --account=project
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=48G
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../results/output/job_%A_%a.out
#SBATCH --error=../results/output/array_job_err_%A_%a.txt
#SBATCH --array=0-2

env="pursuit"
num_agents=8
algo="rmappo" #"mappo" "ippo"
exp="check"
seed=1
case $SLURM_ARRAY_TASK_ID in
   0)   obs_range=7;;
   1)   obs_range=10;;
   2)   obs_range=14;;

esac

SLURM_CPUS_PER_TASK=42
srun --cpus-per-task=$SLURM_CPUS_PER_TASK singularity run \
--rocm -B $SCRATCH:$SCRATCH \
/scratch/project/docker/mujo_gfoot_env_v2.sif \
/bin/sh -c \
"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/partial_marl/.mujoco/mujoco210/bin; \
export PYTHONUSERBASE=/scratch/project/venv_pkgs/mujo_gfoot_env_v2; \
python ./train/train_pursuit.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --num_agents ${num_agents} \
    --n_training_threads 1 --n_rollout_threads 100 --num_mini_batch 1 --episode_length 100 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "partial_marl" \
    --user_name "partial_marl" --use_eval --obs_range ${obs_range}"