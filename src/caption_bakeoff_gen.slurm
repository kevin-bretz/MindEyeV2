#!/bin/bash
# Caption bake-off — Phase 1 (compute node):
#   1. Provision a fresh `vlm_venv` with modern transformers + Qwen2.5-VL deps
#      (idempotent: only installs if vlm_venv/.ok sentinel is missing).
#   2. Generate captions for 50 NSD test images × 6 VLM schemas with
#      Qwen2.5-VL-72B-Instruct-AWQ (schema 0 reuses NSD-GT, no VLM call).
#   3. On success, auto-submit caption_bakeoff_render.slurm (Phase 2).
#
# Self-chaining via afterany: pre-schedules a successor at startup. If we
# finish, the successor self-exits because evals/caption_bakeoff/captions.json
# has all 7 schemas. If we timeout, successor resumes (the script already
# loads existing JSON and only generates missing schemas).
#
# Submit:
#   sbatch src/caption_bakeoff_gen.slurm
# To use 7B fallback (faster, smaller download):
#   sbatch --export=ALL,VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
#          src/caption_bakeoff_gen.slurm
#
#SBATCH --job-name=cn_capgen
#SBATCH --nodes=1
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=03:59:00
#SBATCH -e /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.err
#SBATCH -o /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.out
#SBATCH --no-requeue

set -euo pipefail

module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0
# DO NOT load PyYAML/Python-bundle-PyPI: it injects /easybuild/.../site-packages
# onto PYTHONPATH at provision time, making pip skip transitive deps it sees as
# "already installed". The venv ends up with partial dep graphs (idna, dateutil,
# shellingham etc. missing). Clearing PYTHONPATH below as a safety net.
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
unset PYTHONPATH

cd /home/s4483480/MindEye2/MindEyeV2/src

VLM_VENV="vlm_venv"
VLM_MODEL_ID="${VLM_MODEL_ID:-Qwen/Qwen2.5-VL-72B-Instruct-AWQ}"
N_IMAGES="${N_IMAGES:-50}"
SCRIPT_PATH="/home/s4483480/MindEye2/MindEyeV2/src/caption_bakeoff_gen.slurm"
RENDER_PATH="/home/s4483480/MindEye2/MindEyeV2/src/caption_bakeoff_render.slurm"
OUT_JSON="evals/caption_bakeoff/captions.json"
PHASE2_CSV="tables/caption_bakeoff.csv"

echo "=== Caption bake-off Phase 1 (caption gen) ==="
echo "  VLM model: ${VLM_MODEL_ID}"
echo "  N images:  ${N_IMAGES}"
echo "  Output:    ${OUT_JSON}"

# Idempotency: if Phase-2 CSV already exists, the entire pipeline is past
# this stage; do NOTHING (no successor, no Phase-2 resubmit). Stops the
# fork-bomb pattern where many gen-successors each re-submit Phase 2.
if [ -f "${PHASE2_CSV}" ]; then
    echo "Phase-2 output ${PHASE2_CSV} already exists; nothing for Phase 1 to do, exiting."
    exit 0
fi

# Pre-schedule successor (afterany) so if we timeout mid-download or mid-gen,
# a fresh job picks up via the resumable JSON.
NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
              --export=ALL,VLM_MODEL_ID=${VLM_MODEL_ID},N_IMAGES=${N_IMAGES} \
              "${SCRIPT_PATH}")
echo "Pre-scheduled successor: ${NEXT}"

# --- Stage 1: provision vlm_venv if not ready ---
if [ ! -f "${VLM_VENV}/.ok" ]; then
    echo ""
    echo "--- Stage 1: provisioning ${VLM_VENV} ---"
    if [ ! -d "${VLM_VENV}" ]; then
        python -m venv "${VLM_VENV}"
        echo "  created venv at ${VLM_VENV}"
    fi
    # shellcheck disable=SC1090
    source "${VLM_VENV}/bin/activate"
    pip install --upgrade pip wheel setuptools
    # Torch matching CUDA 12.1 (re-use the wheel index used by fmri venv)
    pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121
    # Pin transformers==4.51.3: autoawq 0.2.9's own deprecation banner says
    # this is the last tested transformers version. 4.52+ removed
    # PytorchGELUTanh which autoawq imports unconditionally.
    pip install --no-cache-dir \
        "transformers==4.51.3" \
        "accelerate>=0.34.0" \
        "datasets" \
        "pandas" \
        "qwen-vl-utils[decord]" \
        "pillow>=10.0" \
        "tqdm" \
        "autoawq>=0.2.6"
    touch "${VLM_VENV}/.ok"
    echo "  marked ${VLM_VENV} ready"
    deactivate
else
    echo "  ${VLM_VENV} already provisioned (.ok sentinel present)"
fi

# --- Stage 2: generate captions ---
echo ""
echo "--- Stage 2: caption_bakeoff_generate.py ---"
# shellcheck disable=SC1090
source "${VLM_VENV}/bin/activate"
echo "  python: $(which python) ($(python -V))"
# Verify the full transformers + autoawq import chain works. If not, pin
# transformers==4.51.3 + tokenizers<0.22 (both required by autoawq 0.2.9 per
# its own deprecation banner: "last tested w/ Transformers 4.51.3").
if ! python -c "from transformers.activations import PytorchGELUTanh; from awq import AutoAWQForCausalLM" 2>/dev/null; then
    echo "  pinning transformers==4.51.3 + tokenizers<0.22 for autoawq 0.2.9"
    pip install --force-reinstall --no-deps \
        "transformers==4.51.3" "tokenizers>=0.21,<0.22" >/dev/null
fi
python -c "import torch, transformers, tokenizers, datasets, awq, qwen_vl_utils; print(f'  torch {torch.__version__}, transformers {transformers.__version__}, tokenizers {tokenizers.__version__}, datasets {datasets.__version__}, awq {awq.__version__}')"
python -c "from transformers.activations import PytorchGELUTanh; from awq import AutoAWQForCausalLM; print('  full import chain OK')"

# Stage HF cache locally to avoid re-downloading per-job
export HF_HOME="${HF_HOME:-/zfsstore/user/s4483480/MindEye2/MindEyeV2/.cache/huggingface}"
mkdir -p "${HF_HOME}"
echo "  HF_HOME=${HF_HOME}"

python caption_bakeoff_generate.py \
    --n_images="${N_IMAGES}" \
    --model_id="${VLM_MODEL_ID}"

deactivate

# --- Stage 3: chain Phase 2 if all schemas present ---
if [ -f "${OUT_JSON}" ]; then
    HAVE_ALL=$(python - <<EOF
import json
need = {"0_nsd_gt", "1_short", "2_dense", "3_tags",
        "4_sent_tags", "5_style", "6_positional"}
try:
    with open("${OUT_JSON}") as f:
        c = json.load(f)
    n = ${N_IMAGES}
    missing = [k for k in need if k not in c or len(c[k]) != n]
    print("YES" if not missing else f"NO:{missing}")
except Exception as e:
    print(f"NO:err:{e}")
EOF
    )
    echo "  schema-completeness check: ${HAVE_ALL}"
    if [[ "${HAVE_ALL}" == "YES" ]]; then
        echo "=== All schemas complete; submitting Phase 2 (render+eval) ==="
        RENDER_JID=$(sbatch --parsable "${RENDER_PATH}")
        echo "  submitted Phase 2: ${RENDER_JID}"
        echo "  (Successor ${NEXT} will self-exit because all schemas present.)"
    else
        echo "=== Some schemas missing; successor ${NEXT} will resume. ==="
    fi
else
    echo "=== ${OUT_JSON} missing; successor ${NEXT} will resume. ==="
fi
