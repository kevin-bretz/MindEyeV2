#!/bin/bash
#SBATCH --job-name=mindeye_finetune
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1    # Request n A100 GPU's
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time 130:00:00      # Adjust based on your needs (1 session ~few hours, 40 sessions ~1 day)
#SBATCH -e slurms/%j.err     # Error log (create 'slurms' folder first)
#SBATCH -o slurms/%j.out     # Output log
#SBATCH --no-requeue

# Load required modules
module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0 
module load PyYAML/6.0.1-GCCcore-13.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Navigate to project directory
cd /home/s4483480/MindEye2/MindEyeV2/src

# Activate virtual environment
source fmri/bin/activate

# Convert notebook to Python script (only needs to be done once)
# jupyter nbconvert Train.ipynb --to python

# Set environment variables
export NUM_GPUS=1
export BATCH_SIZE=24  # Per-GPU batch size
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Random port to avoid conflicts
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))

# Model configuration
model_name="pretrained_subj07_1024hid_40sess_low_300ep"  # Change as needed
num_sessions=40  # Change to 40 for full training

echo "Model name: ${model_name}"
echo "Number of sessions: ${num_sessions}"
echo "Batch size: ${BATCH_SIZE}"

# Run fine-tuning
accelerate launch \
  --num_processes=${NUM_GPUS} \
  --mixed_precision=fp16 \
  Train.py \
  --model_name=${model_name} \
  --multi_subject \
  --subj=7 \
  --batch_size=${BATCH_SIZE} \
  --max_lr=3e-4 \
  --mixup_pct=.33 \
  --num_epochs=300 \
  --use_prior \
  --prior_scale=30 \
  --clip_scale=1 \
  --blurry_recon \
  --blur_scale=.5 \
  --no-use_image_aug \
  --n_blocks=4 \
  --hidden_dim=1024 \
  --num_sessions=${num_sessions} \
  --ckpt_interval=50 \
  --ckpt_saving \


echo "Training complete! Model saved to: ../train_logs/${model_name}"

#   --multisubject_ckpt=/home/s4483480/MindEye2/MindEyeV2/src/models/train_logs/multisubject_subj01_1024hid_nolow_300ep