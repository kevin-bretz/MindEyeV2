#!/bin/bash
# Generic per-subject final_evals stage. Takes MODEL_NAME and SUBJ (default 1)
# as env vars, parameterized on SUBJ so the subj02 timepoint sweeps can chain
# into it.
#
# Submit example:
#   sbatch --export=ALL,MODEL_NAME=subj02_v4a_tp15,SUBJ=2,NO_BLUR_AVG=0 \
#          src/final_evals_subj_sweep.slurm
#SBATCH --job-name=evalsub
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=01:00:00
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

if [ -z "${MODEL_NAME:-}" ]; then
    echo "ERROR: MODEL_NAME must be set via --export=ALL,MODEL_NAME=..." >&2
    exit 1
fi

SUBJ="${SUBJ:-1}"
DATA_PATH="/home/s4483480/MindEye2/MindEyeV2/src"
CACHE_DIR="/home/s4483480/MindEye2/MindEyeV2/src"

ALL_RECONS_PATH="evals/${MODEL_NAME}/${MODEL_NAME}_all_enhancedrecons.pt"
if [ ! -f "$ALL_RECONS_PATH" ]; then
    echo "Enhanced recons missing, falling back to unrefined"
    ALL_RECONS_PATH="evals/${MODEL_NAME}/${MODEL_NAME}_all_recons.pt"
fi

NO_BLUR_AVG="${NO_BLUR_AVG:-0}"
EXTRA_ARGS=()
EVAL_MODEL_NAME="${MODEL_NAME}"
if [ "$NO_BLUR_AVG" = "1" ]; then
    EXTRA_ARGS+=(--no_blur_avg)
    EVAL_MODEL_NAME="${MODEL_NAME}_noblur"
    mkdir -p "evals/${EVAL_MODEL_NAME}"
    for f in all_recons all_enhancedrecons all_blurryrecons all_clipvoxels \
             all_backbones all_prior_out all_predcaptions; do
        src="$(pwd)/evals/${MODEL_NAME}/${MODEL_NAME}_${f}.pt"
        dst="evals/${EVAL_MODEL_NAME}/${EVAL_MODEL_NAME}_${f}.pt"
        [ -f "$src" ] && [ ! -e "$dst" ] && ln -sf "$src" "$dst"
    done
    ALL_RECONS_PATH="evals/${EVAL_MODEL_NAME}/${EVAL_MODEL_NAME}_all_enhancedrecons.pt"
fi

echo "=== Sweep final eval: ${EVAL_MODEL_NAME} (subj=${SUBJ}) ==="
echo "  Recons:        ${ALL_RECONS_PATH}"
echo "  No-blur-avg:   ${NO_BLUR_AVG}"

python final_evaluations.py \
  --model_name=${EVAL_MODEL_NAME} \
  --all_recons_path=${ALL_RECONS_PATH} \
  --data_path=${DATA_PATH} \
  --cache_dir=${CACHE_DIR} \
  --subj=${SUBJ} \
  --seed=42 \
  "${EXTRA_ARGS[@]}"

echo "=== Done. Results: tables/${EVAL_MODEL_NAME}*.csv ==="
