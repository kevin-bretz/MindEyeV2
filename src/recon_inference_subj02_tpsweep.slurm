#!/bin/bash
# Subj02 T_unclip sweep: vary --img2img_timepoint at the unCLIP stage
# (Brain-IT-style init from blurry recons). Each tp value is a separate
# parallel submission with its own model name so outputs don't clobber.
#
# Submit via:
#   for tp in 10 15 20 25 30 35; do
#     sbatch --export=ALL,IMG2IMG_TIMEPOINT=$tp \
#            src/recon_inference_subj02_tpsweep.slurm
#   done
#
# Stages 1+2 here; final_evals (with-mix + no-blur) auto-chained via afterok.
#
#SBATCH --job-name=s2_tp
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=03:59:00
#SBATCH -e /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.err
#SBATCH -o /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.out
#SBATCH --no-requeue

set -e

module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

cd /home/s4483480/MindEye2/MindEyeV2/src
source fmri/bin/activate

IMG2IMG_TIMEPOINT="${IMG2IMG_TIMEPOINT:?must set IMG2IMG_TIMEPOINT}"
SUBJ=2
DATA_PATH="/home/s4483480/MindEye2/MindEyeV2/src"
CACHE_DIR="/home/s4483480/MindEye2/MindEyeV2/src"
SRC_MODEL="finetuned_subj02_1sess_1024hid_low"
MODEL_NAME="subj02_v4a_tp${IMG2IMG_TIMEPOINT}"

# Symlink each sweep's model dir to the underlying finetune's last.pth, so
# recon_inference.py finds the brain weights regardless of the suffix.
ALIAS_DIR="/home/s4483480/MindEye2/MindEyeV2/train_logs/${MODEL_NAME}"
mkdir -p "${ALIAS_DIR}"
if [ ! -e "${ALIAS_DIR}/last.pth" ]; then
    ln -sf "../${SRC_MODEL}/last.pth" "${ALIAS_DIR}/last.pth"
fi

echo "=== Subj02 T_unclip sweep: tp=${IMG2IMG_TIMEPOINT}, model=${MODEL_NAME} ==="
hostname

echo "--- Stage 1/2: recon_inference.py (img2img init, tp=${IMG2IMG_TIMEPOINT}) ---"
python recon_inference.py \
  --model_name=${MODEL_NAME} \
  --data_path=${DATA_PATH} \
  --cache_dir=${CACHE_DIR} \
  --subj=${SUBJ} \
  --n_blocks=4 \
  --hidden_dim=1024 \
  --blurry_recon \
  --new_test \
  --use_img2img_init \
  --img2img_timepoint=${IMG2IMG_TIMEPOINT}
echo "--- Stage 1/2 complete ---"

echo "--- Stage 2/2: enhanced_recon_inference.py ---"
python enhanced_recon_inference.py \
  --model_name=${MODEL_NAME} \
  --subj=${SUBJ}
echo "--- Stage 2/2 complete ---"

echo "--- Chaining final_evals (with-mix + no-blur-avg) ---"
EVAL_MIX=$(sbatch --parsable \
                  --dependency=afterok:${SLURM_JOB_ID} \
                  --export=ALL,MODEL_NAME=${MODEL_NAME},SUBJ=${SUBJ},NO_BLUR_AVG=0 \
                  /home/s4483480/MindEye2/MindEyeV2/src/final_evals_subj_sweep.slurm)
EVAL_NOMIX=$(sbatch --parsable \
                    --dependency=afterok:${SLURM_JOB_ID} \
                    --export=ALL,MODEL_NAME=${MODEL_NAME},SUBJ=${SUBJ},NO_BLUR_AVG=1 \
                    /home/s4483480/MindEye2/MindEyeV2/src/final_evals_subj_sweep.slurm)
echo "  with-mix:    ${EVAL_MIX}"
echo "  no-blur-avg: ${EVAL_NOMIX}"

# If this is the tp=15 run, also chain the SDXL sweep (3 jobs) since they
# reuse this run's Stage-1 outputs.
if [ "${IMG2IMG_TIMEPOINT}" = "15" ]; then
    echo "--- tp=15: chaining T_sdxl sweep (sdxl8/sdxl18/sdxl23) ---"
    for sdxl in 8 18 23; do
        SDXL_JID=$(sbatch --parsable \
                          --dependency=afterok:${SLURM_JOB_ID} \
                          --export=ALL,IMG2IMG_TIMEPOINT_SDXL=${sdxl} \
                          /home/s4483480/MindEye2/MindEyeV2/src/recon_inference_subj02_sdxlsweep.slurm)
        echo "  sdxl${sdxl}: ${SDXL_JID}"
    done
fi

echo "=== tp=${IMG2IMG_TIMEPOINT} stages 1+2 done. ==="
