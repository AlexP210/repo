#!/bin/bash
#SBATCH --account=def-rhinehar
#SBATCH --job-name=repo-training
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=/dev/null
#SBATCH --error=/home/alexpl/scratch/outputs/slurm/%A_%a.out

module load python mujoco opencv hdf5
export MUJOCO_GL=egl
export WANDB_API_KEY=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
export WANDB_DIR=/home/alexpl/scratch/repo_training

export DATA_DIR="/home/alexpl/project/def-rhinehar/alexpl/TDMPC2-Data"

source /home/alexpl/envs/tsd-repo/bin/activate

TASK_NAMES=("cheetah-jump" "walker-walk" "cartpole-balance")
USE_TSD_ENCODER_VALUES=("True" "False")
TRAIN_OFFLINE_VALUES=("False" "True")
SEED_VALUES=(1)

USE_TSD_ENCODER_VALUE_IDX=$(( SLURM_ARRAY_TASK_ID % ${#USE_TSD_ENCODER_VALUES[@]} ))
TRAIN_OFFLINE_VALUE_IDX=$(( SLURM_ARRAY_TASK_ID / ${#USE_TSD_ENCODER_VALUES[@]} % ${#TRAIN_OFFLINE_VALUES[@]} ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID / (${#USE_TSD_ENCODER_VALUES[@]} * ${#TRAIN_OFFLINE_VALUES[@]}) % ${#TASK_NAMES[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / (${#USE_TSD_ENCODER_VALUES[@]} * ${#TRAIN_OFFLINE_VALUES[@]} * ${#TASK_NAMES[@]}) ))

CURRENT_TASK_NAME=${TASK_NAMES[$TASK_IDX]}
CURRENT_SEED_VALUE=${SEED_VALUES[$SEED_IDX]}
CURRENT_USE_TSD_ENCODER_VALUE=${USE_TSD_ENCODER_VALUES[$USE_TSD_ENCODER_VALUE_IDX]}
CURRENT_TRAIN_OFFLINE_VALUE=${TRAIN_OFFLINE_VALUES[$TRAIN_OFFLINE_VALUE_IDX]}

cd /home/alexpl/tsd-project/agents/repo/repo/experiments
WANDB_MODE=offline python train_repo.py \
--env_id=dmc-${CURRENT_TASK_NAME} \
--seed=${CURRENT_SEED_VALUE} \
--use_tsd_encoder=${CURRENT_USE_TSD_ENCODER_VALUE} \
--train_offline=${CURRENT_TRAIN_OFFLINE_VALUE} \
--expr_name=TASK${CURRENT_TASK_NAME}_ENCODER${CURRENT_USE_TSD_ENCODER_VALUE}_OFFLINE${CURRENT_TRAIN_OFFLINE_VALUE}_SEED${CURRENT_SEED_VALUE}