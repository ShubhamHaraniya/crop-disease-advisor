#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh — Create venv and install all Phase 1 or Phase 2 dependencies
# Usage: bash setup_env.sh [phase1|phase2]
# ─────────────────────────────────────────────────────────────────────────────
set -e

PHASE="${1:-phase1}"
VENV_DIR="./venv"

echo "════════════════════════════════════════"
echo "  Crop Disease Advisor — Environment Setup"
echo "  Phase : $PHASE"
echo "════════════════════════════════════════"

# ── Python check ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3.10 || command -v python3 || echo "")
if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.10 not found. Install it first."
    exit 1
fi
echo "✓  Found Python: $($PYTHON --version)"

# ── CUDA check ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "✓  GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    CUDA_AVAILABLE=true
else
    echo "⚠  No GPU detected. PyTorch will use CPU."
    CUDA_AVAILABLE=false
fi

# ── Create venv ───────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Creating virtual environment at $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── Install PyTorch ───────────────────────────────────────────────────────────
echo ""
echo "[2/3] Installing PyTorch 2.1.0..."
if [ "$CUDA_AVAILABLE" = true ]; then
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118 -q
else
    pip install torch==2.1.0+cpu torchvision==0.16.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── Install project deps ──────────────────────────────────────────────────────
echo ""
echo "[3/3] Installing $PHASE dependencies..."
if [ "$PHASE" = "phase2" ]; then
    pip install -r requirements_phase2.txt -q
else
    pip install -r requirements_phase1.txt -q
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "Running verification..."
python verify_env.py

echo ""
echo "════════════════════════════════════════"
echo "  ✅ Environment ready!"
echo "  Activate with: source $VENV_DIR/bin/activate"
echo "════════════════════════════════════════"
