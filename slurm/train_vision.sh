#!/bin/bash
#SBATCH --job-name=crop_vision_train
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/vision_%j.out
#SBATCH --error=logs/vision_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@iitj.ac.in

# ── Banner ────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  Crop Disease Vision Training"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  GPU      : $CUDA_VISIBLE_DEVICES"
echo "  Started  : $(date)"
echo "════════════════════════════════════════════"

# ── Step 1: Ensure Python 3.10 libraries are loaded ───────────────────────────
export LD_LIBRARY_PATH="/opt/ohpc/apps/python/3.10.pytorch/lib:${LD_LIBRARY_PATH:-}"

VENV_PATH="${HOME}/MLOps_Project/crop-disease-advisor/venv"
# The correct Python 3.10 + PyTorch binary (hardcoded for this HPC cluster)
MODULE_PYTHON="/opt/ohpc/apps/python/3.10.pytorch/bin/python3"

if [ -d "$VENV_PATH" ]; then
    echo "  Activating venv: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    PYTHON_BIN="$VENV_PATH/bin/python3"
elif [ -x "$MODULE_PYTHON" ]; then
    echo "  No venv found. Creating one with module Python..."
    "$MODULE_PYTHON" -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    PYTHON_BIN="$VENV_PATH/bin/python3"
    echo "  Installing dependencies into fresh venv..."
    "$PYTHON_BIN" -m pip install --quiet --upgrade pip
    "$PYTHON_BIN" -m pip install --quiet \
        torch torchvision --index-url https://download.pytorch.org/whl/cu118
    "$PYTHON_BIN" -m pip install --quiet \
        timm albumentations scikit-learn pyyaml wandb pillow tqdm matplotlib
else
    echo "❌ Cannot find Python 3.10. Looked at: $MODULE_PYTHON"
    echo "   Please run 'bash setup_env.sh phase1' once from an interactive session."
    exit 1
fi

echo "  Python   : $PYTHON_BIN"
echo "  Version  : $($PYTHON_BIN --version 2>&1)"

# ── Step 2: Quick sanity check ────────────────────────────────────────────────
if ! "$PYTHON_BIN" -c "import numpy, torch, timm, sklearn, yaml" 2>/dev/null; then
    echo "❌ Core packages missing from venv. Installing..."
    "$PYTHON_BIN" -m pip install --quiet \
        numpy torch torchvision timm albumentations \
        scikit-learn pyyaml wandb pillow tqdm matplotlib
fi
echo "  ✓ Dependencies OK"

# ── Step 3: Move to project root ──────────────────────────────────────────────
cd "${HOME}/MLOps_Project/crop-disease-advisor" || exit 1
mkdir -p logs models/vision notebooks/sample_augmentations data/processed

# ── W&B ───────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="crop-disease-advisor"

# ── Step 4: Preprocess ────────────────────────────────────────────────────────
echo ""
echo "[1/2] Preprocessing data..."
"$PYTHON_BIN" src/vision/preprocess.py \
    --raw_dir    data/raw/plantvillage \
    --output_dir data/processed \
    --aug_dir    notebooks/sample_augmentations

PREPROCESS_EXIT=$?
if [ $PREPROCESS_EXIT -ne 0 ]; then
    echo "❌ Preprocessing failed (exit $PREPROCESS_EXIT)."
    exit $PREPROCESS_EXIT
fi

# ── Step 5: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Starting EfficientNet-B4 training..."
"$PYTHON_BIN" src/vision/train.py --config configs/vision_config.yaml

TRAIN_EXIT=$?
echo ""
echo "════════════════════════════════════════════"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "  ✅ Training complete: $(date)"
    echo "  Checkpoint: models/vision/efficientnet_b4_best.pt"
else
    echo "  ❌ Training failed (exit $TRAIN_EXIT)."
fi
echo "════════════════════════════════════════════"
exit $TRAIN_EXIT
