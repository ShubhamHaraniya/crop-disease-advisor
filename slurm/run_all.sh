#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Submit Phase 1 and Phase 2 SLURM jobs in sequence
# Phase 2 is submitted as a dependency of Phase 1 (runs after Phase 1 finishes)
# Usage: bash slurm/run_all.sh [phase1|phase2|all]
# ─────────────────────────────────────────────────────────────────────────────
set -e

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════"
echo "  Crop Disease Advisor — SLURM Job Submission"
echo "  Mode: $MODE"
echo "════════════════════════════════════════════════"

case "$MODE" in

  phase1)
    echo ""
    echo "[Phase 1] Submitting vision training job..."
    JOB1=$(sbatch --parsable "$SCRIPT_DIR/train_vision.sh")
    echo "  ✓ Job submitted: $JOB1"
    echo ""
    echo "Monitor: squeue -j $JOB1"
    echo "Logs   : tail -f logs/vision_${JOB1}.out"
    ;;

  phase2)
    echo ""
    echo "[Phase 2] Submitting LLM training job..."
    JOB2=$(sbatch --parsable "$SCRIPT_DIR/train_llm.sh")
    echo "  ✓ Job submitted: $JOB2"
    echo ""
    echo "Monitor: squeue -j $JOB2"
    echo "Logs   : tail -f logs/llm_${JOB2}.out"
    ;;

  all)
    echo ""
    echo "[1/3] Submitting Phase 1 — Vision training..."
    JOB1=$(sbatch --parsable "$SCRIPT_DIR/train_vision.sh")
    echo "  ✓ Phase 1 job: $JOB1"

    echo ""
    echo "[2/3] Submitting Phase 2 — LLM training (after job $JOB1)..."
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$SCRIPT_DIR/train_llm.sh")
    echo "  ✓ Phase 2 job: $JOB2 (will start after $JOB1 succeeds)"

    echo ""
    echo "[3/3] Submitting Evaluation + Registration (after job $JOB1)..."
    EVAL_SCRIPT=$(mktemp /tmp/eval_XXXXXX.sh)
    cat > "$EVAL_SCRIPT" << 'EOFINNER'
#!/bin/bash
#SBATCH --job-name=crop_eval
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.out

source "${HOME}/MLOps_Project/crop-disease-advisor/venv/bin/activate"
cd "${HOME}/MLOps_Project/crop-disease-advisor"

python3 src/vision/evaluate.py --wandb
python3 pipelines/register_model.py \
    --stage staging \
    --hf_username "${HF_USERNAME}" \
    --hf_token    "${HF_TOKEN}"
EOFINNER
    chmod +x "$EVAL_SCRIPT"
    JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 "$EVAL_SCRIPT")
    echo "  ✓ Evaluation job: $JOB3"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Jobs submitted:"
    echo "    Phase 1 (vision train) : $JOB1"
    echo "    Phase 2 (LLM train)    : $JOB2  → runs after $JOB1"
    echo "    Evaluation + Registry  : $JOB3  → runs after $JOB1"
    echo ""
    echo "  Monitor all: squeue -u \$USER"
    echo "  Cancel all : scancel $JOB1 $JOB2 $JOB3"
    echo "════════════════════════════════════════════════"
    ;;

  *)
    echo "Usage: bash $0 [phase1|phase2|all]"
    exit 1
    ;;
esac
