#!/bin/bash
# Publish trained MindEye2 checkpoints to the Hugging Face Hub.
#
# For each model listed in CKPT_MODELS, this loads ../train_logs/<name>/last.pth,
# strips the optimizer state (keeps only `model_state_dict` -> inference-only,
# ~2-4 GB instead of ~8 GB, weights only, NO NSD data), then creates/updates the
# HF model repo and uploads each as <name>/last.pth alongside a model card.
#
# REQUIRES a Hugging Face *WRITE* token:
#   1. Create one at  https://huggingface.co/settings/tokens  (role: Write)
#   2. On the cluster (a login node is fine for this one-off auth step):
#        huggingface-cli login          # paste the token
#      # or:  export HF_TOKEN=hf_xxxxxxxx
#
# Submit:   sbatch src/upload_hf_checkpoint.slurm
# Override: sbatch --export=ALL,HF_REPO=kevin-bretz/mindeye2-subj02,\
#                  CKPT_MODELS="finetuned_subj02_1sess_1024hid_low multisubject_excludingsubj02" \
#                  src/upload_hf_checkpoint.slurm
#
#SBATCH --job-name=hf_upload
#SBATCH --nodes=1
#SBATCH --partition=cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH -e /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.err
#SBATCH -o /home/s4483480/MindEye2/MindEyeV2/src/slurms/%j.out
#SBATCH --no-requeue

set -euo pipefail

module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0
module load PyYAML/6.0.1-GCCcore-13.2.0   # provides `yaml` for huggingface_hub (not pip-installed in the fmri venv)
module load CUDA/12.1.1 2>/dev/null || echo "(CUDA module unavailable; CPU-only upload is fine)"

cd /home/s4483480/MindEye2/MindEyeV2/src
source fmri/bin/activate

HF_REPO="${HF_REPO:-kevin-bretz/mindeye2-subj02}"
CKPT_MODELS="${CKPT_MODELS:-finetuned_subj02_1sess_1024hid_low multisubject_excludingsubj02}"
STAGE="hf_upload"

python -c "import huggingface_hub" 2>/dev/null || pip install -q "huggingface_hub>=0.23"

# --- Preflight: require a usable token (write scope is needed for create_repo) ---
if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "ERROR: no Hugging Face token found on this node." >&2
    echo "  Create a WRITE token at https://huggingface.co/settings/tokens," >&2
    echo "  then run:  huggingface-cli login   (or: export HF_TOKEN=hf_xxx)" >&2
    exit 1
fi

mkdir -p "$STAGE"
cp MODEL_CARD.md "$STAGE/README.md"

# --- Strip each checkpoint to inference-only weights ---
for M in $CKPT_MODELS; do
    SRC="../train_logs/${M}/last.pth"
    if [ ! -f "$SRC" ]; then echo "WARN: $SRC not found, skipping"; continue; fi
    mkdir -p "$STAGE/$M"
    echo "--- stripping ${M} ---"
    python - "$SRC" "$STAGE/$M/last.pth" <<'PY'
import sys, os, torch
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
torch.save({"model_state_dict": sd}, dst)
print(f"  {os.path.getsize(src)/1e9:.1f} GB -> {os.path.getsize(dst)/1e9:.1f} GB")
PY
done

# --- Create repo (idempotent) + upload the whole staged folder ---
echo "--- uploading to ${HF_REPO} ---"
python - "$HF_REPO" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                  commit_message="Add MindEye2 Subject-2 checkpoints (inference-only)")
print(f"Done -> https://huggingface.co/{repo}")
PY

echo "Uploaded to https://huggingface.co/${HF_REPO}"
