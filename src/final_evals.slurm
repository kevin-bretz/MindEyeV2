#!/bin/bash
#SBATCH --job-name=mindeye_eval
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
cd /home/s4483480/MindEye2/MindEyeV2/src

# Activate virtual environment
source fmri/bin/activate

# Model configuration
model_name="finetuned_subj07_1024hid_40sess_low_150ep"
# Use enhanced reconstructions if available, otherwise use regular recons
all_recons_path="evals/${model_name}/${model_name}_all_enhancedrecons.pt"
if [ ! -f "$all_recons_path" ]; then
    echo "Enhanced recons not found, using regular reconstructions"
    all_recons_path="evals/${model_name}/${model_name}_all_recons.pt"
fi

data_path="/home/s4483480/MindEye2/MindEyeV2/src"
cache_dir="/home/s4483480/MindEye2/MindEyeV2/src"
subj=7

echo "Running final evaluations on model: ${model_name}"
echo "Reconstructions path: ${all_recons_path}"
echo "Subject: ${subj}"

# Run final evaluations
python final_evaluations.py \
  --model_name=${model_name} \
  --all_recons_path=${all_recons_path} \
  --data_path=${data_path} \
  --cache_dir=${cache_dir} \
  --subj=${subj} \
  --seed=42

echo "Evaluation complete! Results saved to tables/${model_name}.csv"
