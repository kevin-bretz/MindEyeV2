#!/bin/bash
#SBATCH --job-name=boi_latent
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=04:00:00
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --no-requeue

module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

cd /home/s4556534/mindEye2/MindEye2/MindEyeV2/src
source fmri/bin/activate

model_name="finetune_subj02_150ep"
data_path="/home/s4556534/mindEye2/MindEye2/MindEyeV2/src"
cache_dir="/home/s4556534/mindEye2/MindEye2/MindEyeV2/src"
subj=2
n_candidates=8
n_iterations=3

mkdir -p slurms

echo "Started:     $(date)"
echo "Model:       ${model_name}"
echo "Subject:     ${subj}"
echo "Candidates:  ${n_candidates}"
echo "Iterations:  ${n_iterations}"

python brain_optimized_inference_latent.py \
    --model_name=${model_name} \
    --data_path=${data_path} \
    --cache_dir=${cache_dir} \
    --subj=${subj} \
    --new_test \
    --n_candidates=${n_candidates} \
    --n_iterations=${n_iterations} \
    --convergence_tol=0.05

echo "Finished or timed out: $(date)"

# Check how many images are done and resubmit if not complete
python -c "
import torch, sys, os
f = 'evals/${model_name}/${model_name}_all_brain_opt_recons_v2.pt'
if not os.path.exists(f):
    print('No checkpoint found yet')
    sys.exit(1)
r = torch.load(f, map_location='cpu')
n = len(r)
print(f'Progress: {n}/1000 images done')
sys.exit(0 if n >= 1000 else 1)
"

if [ $? -ne 0 ]; then
    echo "Not done yet — resubmitting..."
    sbatch /home/s4556534/mindEye2/MindEye2/MindEyeV2/src/brain_opt_inference.slurm
else
    echo "All 1000 images complete! Run evaluation next."
fi