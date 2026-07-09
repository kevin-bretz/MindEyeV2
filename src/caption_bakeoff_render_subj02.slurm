#!/bin/bash
# VLM caption ablation for subj02 — re-uses the existing
# evals/caption_bakeoff/captions.json (subject-agnostic: captions are about the
# 50 NSD test images themselves, not the brain). The img2img *source* recons
# DO depend on the subject — for subj02 we use the baseline recons by default,
# overridable via SOURCE_MODEL env var.
#
# Writes per-schema refined images and the leaderboard CSV under a subj02-
# specific name so it doesn't clobber the subj01 caption bake-off CSV.
#
# Submit with:
#   sbatch --export=ALL,SOURCE_MODEL=finetuned_subj02_1sess_1024hid_low,SUBJ=2 \
#          src/caption_bakeoff_render_subj02.slurm
#
#SBATCH --job-name=s2_capr
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

SOURCE_MODEL="${SOURCE_MODEL:-finetuned_subj02_1sess_1024hid_low}"
SUBJ="${SUBJ:-2}"
N_IMAGES="${N_IMAGES:-50}"
SCRIPT_PATH="/home/s4483480/MindEye2/MindEyeV2/src/caption_bakeoff_render_subj02.slurm"

CSV_OUT="tables/caption_bakeoff_subj02.csv"
JSON_IN="evals/caption_bakeoff/captions.json"

echo "=== VLM caption ablation (subj02 render+eval) ==="
echo "  Captions:                ${JSON_IN}"
echo "  Source recons (subj02):  ${SOURCE_MODEL}"
echo "  Output CSV:              ${CSV_OUT}"

if [ ! -f "${JSON_IN}" ]; then
    echo "ERROR: ${JSON_IN} not found. Run caption_bakeoff_gen.slurm first." >&2
    exit 1
fi

if [ ! -f "evals/${SOURCE_MODEL}/${SOURCE_MODEL}_all_recons.pt" ]; then
    echo "ERROR: missing source recons evals/${SOURCE_MODEL}/${SOURCE_MODEL}_all_recons.pt" >&2
    exit 1
fi

if [ -f "${CSV_OUT}" ]; then
    echo "Output CSV already exists; exiting (delete to re-run)."
    cat "${CSV_OUT}"
    exit 0
fi

# Pre-schedule successor for resume on timeout.
NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
              --export=ALL,SOURCE_MODEL=${SOURCE_MODEL},SUBJ=${SUBJ},N_IMAGES=${N_IMAGES} \
              "${SCRIPT_PATH}")
echo "Pre-scheduled successor: ${NEXT}"

# The render script reads evals/all_images.pt for GT — same blob for all subjs
# (NSD shared-1000 test images), so no subj-specific arg needed there.
# But the per-schema refined_*.pt cache would collide with subj01's run, so
# write into a subj02-namespaced output_dir.
SUBJ_OUTDIR="evals/caption_bakeoff_subj02"
mkdir -p "${SUBJ_OUTDIR}"

python caption_bakeoff_render.py \
    --captions_path="${JSON_IN}" \
    --source_model="${SOURCE_MODEL}" \
    --n_images="${N_IMAGES}" \
    --output_dir="${SUBJ_OUTDIR}" \
    --csv_path="${CSV_OUT}"

if [ -f "${CSV_OUT}" ]; then
    echo ""
    echo "=== VLM caption ablation complete. ==="
    cat "${CSV_OUT}"
    echo "(Successor ${NEXT} will self-exit because CSV exists.)"
fi
