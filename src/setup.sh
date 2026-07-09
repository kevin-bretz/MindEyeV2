#!/bin/bash
# MindEye2 setup for the ALICE HPC cluster.
#
# Creates the `fmri` virtualenv inside the current directory and installs all
# pinned dependencies. Safe to re-run — it reuses an existing venv.
#
# Usage (from MindEyeV2/src/):
#   source setup.sh       # keeps the activated venv in your current shell
#   # OR
#   bash   setup.sh       # runs in a subshell; activate afterwards with
#                         #   source fmri/bin/activate
#
# After setup finishes, activate the venv in any new shell via:
#   source /path/to/MindEyeV2/src/fmri/bin/activate

# Detect sourced vs executed so we can `return` from a sourced script and
# `exit` from an executed one. Avoids nuking the user's login shell on error.
(return 0 2>/dev/null) && SOURCED=1 || SOURCED=0
die() { echo "ERROR: $*" >&2; [ "$SOURCED" = "1" ] && return 1 || exit 1; }

# --- 1. Sanity-check the working directory -----------------------------------
if [ ! -f "Train.py" ] || [ ! -d "generative_models" ]; then
    die "run this from MindEyeV2/src/  (got: $(pwd))"
fi

# --- 2. Load ALICE HPC modules -----------------------------------------------
# These are required both for the Python interpreter used to build the venv
# and for CUDA to be visible inside the venv at runtime.
if command -v module >/dev/null 2>&1; then
    echo "[1/4] Loading ALICE modules..."
    module purge
    module load ALICE/default
    module load Python/3.11.5-GCCcore-13.2.0
    module load PyYAML/6.0.1-GCCcore-13.2.0
    module load CUDA/12.1.1
    module load cuDNN/8.9.2.26-CUDA-12.1.1
else
    echo "[1/4] 'module' command not found — assuming Python 3.11 + CUDA 12.1 are already on PATH."
fi

command -v python3.11 >/dev/null 2>&1 || die "python3.11 not found after loading modules."

# --- 3. Create (or reuse) the fmri venv --------------------------------------
if [ -d "fmri" ]; then
    echo "[2/4] Reusing existing fmri/ venv."
else
    echo "[2/4] Creating fmri/ venv with $(python3.11 --version)..."
    python3.11 -m venv fmri || die "venv creation failed"
fi

# shellcheck disable=SC1091
source fmri/bin/activate || die "failed to activate fmri venv"

# --- 4. Install pinned dependencies ------------------------------------------
echo "[3/4] Upgrading pip / wheel / setuptools..."
pip install --upgrade pip wheel >/dev/null
pip install setuptools==69.5.1  >/dev/null

echo "[3/4] Installing pinned requirements (takes several minutes on first run)..."
pip install -r requirements.txt || die "pip install -r requirements.txt failed"

# dalle2-pytorch pins an incompatible torch version, so install its runtime
# deps first and then the package itself with --no-deps.
echo "[3/4] Installing dalle2-pytorch (--no-deps)..."
pip install vector-quantize-pytorch einops-exts resize-right
pip install dalle2-pytorch --no-deps

# --- 5. Smoke test -----------------------------------------------------------
echo "[4/4] Running import smoke test..."
python - <<'PY'
import sys
sys.path.append("generative_models/")
import torch
import sgm  # noqa: F401
print(f"  torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
PY

cat <<EOF

=== Setup complete! ===

The fmri venv lives at: $(pwd)/fmri
Activate it in any new shell with:
    source $(pwd)/fmri/bin/activate

Next steps:
  1. Download the NSD subset with:  python download_data.py
  2. Create a slurms/ directory for SLURM logs:  mkdir -p slurms
  3. Submit a job, e.g.:  sbatch finetune_subj01.slurm
     (edit the slurm file first to point to your own project directory).

EOF
