#!/bin/bash
#SBATCH --job-name=crop_llm_train
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err

# ── Banner ────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo "  Crop Disease LLM QLoRA Training"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  GPU      : $CUDA_VISIBLE_DEVICES"
echo "  Started  : $(date)"
echo "════════════════════════════════════════════"

# ── Step 1: Ensure Python 3.10 libraries are loaded ───────────────────────────
export LD_LIBRARY_PATH="/opt/ohpc/apps/python/3.10.pytorch/lib:${LD_LIBRARY_PATH:-}"

VENV_PATH="${HOME}/MLOps_Project/crop-disease-advisor/venv"
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
    echo "  Installing base PyTorch..."
    "$PYTHON_BIN" -m pip install --quiet --upgrade pip
    "$PYTHON_BIN" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "❌ Cannot find Python 3.10. Looked at: $MODULE_PYTHON"
    exit 1
fi

echo "  Python   : $PYTHON_BIN"
echo "  Version  : $($PYTHON_BIN --version 2>&1)"

# ── Step 2: Ensure Phase 2 Dependencies ────────────────────────────────────────
echo ""
echo "[1/3] Checking LLM dependencies..."
if ! "$PYTHON_BIN" -c "import transformers, peft, datasets, bitsandbytes" 2>/dev/null; then
    echo "  Installing Phase 2 packages (Transformers, PEFT, TRL, BitsAndBytes)..."
    "$PYTHON_BIN" -m pip install --quiet transformers peft accelerate bitsandbytes datasets trl \
        --extra-index-url https://download.pytorch.org/whl/cu118
fi
echo "  ✓ Dependencies OK"

# ── Step 3: Move to project root ──────────────────────────────────────────────
cd "${HOME}/MLOps_Project/crop-disease-advisor" || exit 1
mkdir -p logs models/llm

# ── W&B and HF Login ──────────────────────────────────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="crop-disease-advisor-llm"
export HF_TOKEN="${HF_TOKEN:-}"

# ── Step 4: Generate Synthetic Dataset ────────────────────────────────────────
echo ""
echo "[2/3] Generating instruction dataset (50,000 pairs)..."
if [ ! -f "data/processed/llm_instructions.jsonl" ]; then
    "$PYTHON_BIN" src/llm/generate_dataset.py \
        --n 50000 \
        --output data/processed/llm_instructions.jsonl \
        --seed 42
else
    echo "  ✓ Dataset already exists."
fi

# ── Step 5: Train QLoRA ───────────────────────────────────────────────────────
echo ""
echo "[3/3] Starting QLoRA fine-tuning..."
"$PYTHON_BIN" src/llm/train_qlora.py --config configs/llm_config.yaml

TRAIN_EXIT=$?
echo ""
echo "════════════════════════════════════════════"
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "  ✅ QLoRA Training complete: $(date)"
    echo "  Adapters saved to: models/llm/llama3_qlora_adapter"
else
    echo "  ❌ Training failed (exit $TRAIN_EXIT). See logs/llm_${SLURM_JOB_ID}.err"
fi
echo "════════════════════════════════════════════"
exit $TRAIN_EXIT
