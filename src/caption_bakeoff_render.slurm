#!/bin/bash
# Caption bake-off — Phase 2 (compute node):
#   1. Run SDXL img2img on the same 50 brain-predicted starting images for
#      each of the 7 caption schemas in evals/caption_bakeoff/captions.json.
#   2. Compute the standard MindEye2 metric suite (PixCorr, SSIM, AlexNet5,
#      CLIP, EffNet-B, SwAV) for each schema's refined outputs vs ground
#      truth.
#   3. Write tables/caption_bakeoff.csv (one row per schema, sorted by
#      composite-5 score).
#
# Self-chaining via afterany: if we timeout mid-img2img, the successor picks
# up because each schema's refined_<schema>.pt is saved atomically per-batch.
#
# Submit (auto-submitted by caption_bakeoff_gen.slurm on success):
#   sbatch src/caption_bakeoff_render.slurm
#
#SBATCH --job-name=cn_caprender
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=03:59:00
#SBATCH -e /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.err
#SBATCH -o /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.out
#SBATCH --no-requeue

set -euo pipefail

module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

cd /home/s4483480/MindEye2/MindEyeV2/src
source fmri/bin/activate

SOURCE_MODEL="${SOURCE_MODEL:-finetuned_subj01_40sess_1024hid_low}"
N_IMAGES="${N_IMAGES:-50}"
SCRIPT_PATH="/home/s4483480/MindEye2/MindEyeV2/src/caption_bakeoff_render.slurm"

CSV_OUT="tables/caption_bakeoff.csv"
JSON_IN="evals/caption_bakeoff/captions.json"

echo "=== Caption bake-off Phase 2 (render + eval) ==="
echo "  Captions: ${JSON_IN}"
echo "  Source recons (starting): ${SOURCE_MODEL}"
echo "  Output CSV:               ${CSV_OUT}"

if [ ! -f "${JSON_IN}" ]; then
    echo "ERROR: ${JSON_IN} not found. Run caption_bakeoff_gen.slurm first." >&2
    exit 1
fi

if [ -f "${CSV_OUT}" ]; then
    echo "Output CSV already exists; exiting (delete to re-run)."
    cat "${CSV_OUT}"
    exit 0
fi

# Pre-schedule successor for resume on timeout (each schema's refined_*.pt
# is cached, so this resumes by skipping completed schemas).
NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
              --export=ALL,SOURCE_MODEL=${SOURCE_MODEL},N_IMAGES=${N_IMAGES} \
              "${SCRIPT_PATH}")
echo "Pre-scheduled successor: ${NEXT}"

python caption_bakeoff_render.py \
    --captions_path="${JSON_IN}" \
    --source_model="${SOURCE_MODEL}" \
    --n_images="${N_IMAGES}" \
    --csv_path="${CSV_OUT}"

if [ -f "${CSV_OUT}" ]; then
    echo ""
    echo "=== Phase 2 complete. ==="
    echo "Results:"
    cat "${CSV_OUT}"
    echo ""
    echo "(Successor ${NEXT} will self-exit because CSV exists.)"
fi
