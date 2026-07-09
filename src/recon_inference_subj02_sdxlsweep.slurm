#!/bin/bash
# Subj02 T_sdxl sweep at fixed T_unclip=15. Reuses cached Stage-1 outputs
# from subj02_v4a_tp15 by symlinking under a new model name per T_sdxl value.
# Only re-runs Stage 2 (enhanced SDXL img2img refinement) which is fast.
#
# Auto-submitted by recon_inference_subj02_tpsweep.slurm when tp=15. Can also
# be submitted manually:
#   for sdxl in 8 18 23; do
#     sbatch --export=ALL,IMG2IMG_TIMEPOINT_SDXL=$sdxl \
#            src/recon_inference_subj02_sdxlsweep.slurm
#   done
#
#SBATCH --job-name=s2_sdxl
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=01:30:00
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

IMG2IMG_TIMEPOINT_SDXL="${IMG2IMG_TIMEPOINT_SDXL:?must set IMG2IMG_TIMEPOINT_SDXL}"
SUBJ=2
SRC_MODEL="subj02_v4a_tp15"
MODEL_NAME="subj02_v4a_tp15_sdxl${IMG2IMG_TIMEPOINT_SDXL}"

# Symlink Stage-1 cached outputs from src model into the new model dir.
SRC_DIR="evals/${SRC_MODEL}"
DST_DIR="evals/${MODEL_NAME}"
mkdir -p "${DST_DIR}"
for f in all_recons all_blurryrecons all_clipvoxels all_backbones all_prior_out all_predcaptions; do
    src_abs="$(pwd)/${SRC_DIR}/${SRC_MODEL}_${f}.pt"
    dst="${DST_DIR}/${MODEL_NAME}_${f}.pt"
    if [ -f "$src_abs" ] && [ ! -e "$dst" ]; then
        ln -sf "$src_abs" "$dst"
    fi
done

echo "=== Subj02 T_sdxl sweep: T_unclip=15, T_sdxl=${IMG2IMG_TIMEPOINT_SDXL}, model=${MODEL_NAME} ==="
hostname

echo "--- Stage 2: enhanced_recon_inference.py (img2img_timepoint=${IMG2IMG_TIMEPOINT_SDXL}) ---"
python enhanced_recon_inference.py \
  --model_name=${MODEL_NAME} \
  --subj=${SUBJ} \
  --img2img_timepoint=${IMG2IMG_TIMEPOINT_SDXL}
echo "--- Stage 2 complete ---"

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

echo "=== T_sdxl=${IMG2IMG_TIMEPOINT_SDXL} done. ==="
