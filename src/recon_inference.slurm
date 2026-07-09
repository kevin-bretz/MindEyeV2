#!/bin/bash
#SBATCH --job-name=mindeye_inference
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=04:00:00
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue

# Load required modules
module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0 
module load PyYAML/6.0.1-GCCcore-13.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Navigate to project directory
cd /home/s4556534/mindEye2/MindEye2/MindEyeV2/src

# Activate virtual environment
source fmri/bin/activate

# Model configuration
model_name="finetune_subj02_150ep"
data_path="/home/s4556534/mindEye2/MindEye2/MindEyeV2/src"
cache_dir="/home/s4556534/mindEye2/MindEye2/MindEyeV2/src"

echo "Running inference on model: ${model_name}"
echo "Subject: 2"
echo "Architecture: hidden_dim=1024, n_blocks=4"

# Run regular reconstruction inference
python recon_inference.py \
  --model_name=${model_name} \
  --data_path=${data_path} \
  --cache_dir=${cache_dir} \
  --subj=2 \
  --n_blocks=4 \
  --hidden_dim=1024 \
  --blurry_recon \
  --new_test

echo "Regular reconstruction complete!"

# Run enhanced reconstruction (refinement stage)
python enhanced_recon_inference.py \
  --model_name=${model_name} \
  --subj=2

echo "Enhanced reconstruction complete!"
echo "Results saved to: evals/${model_name}/"