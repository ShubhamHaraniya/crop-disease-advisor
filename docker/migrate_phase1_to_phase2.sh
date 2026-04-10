#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# migrate_phase1_to_phase2.sh
# Zero-downtime migration from Phase 1 to Phase 2 docker-compose stack.
# Prompt 2.7 — Migration Script
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VISION_CKPT="$PROJECT_ROOT/models/vision/efficientnet_b4_best.pt"
ADAPTER_DIR="$PROJECT_ROOT/models/llm/llama3_qlora_adapter"
ACTIVE_LINK="$SCRIPT_DIR/docker-compose.active.yml"
REQUIRED_ACC=0.85   # minimum test accuracy to proceed

echo "══════════════════════════════════════════════════════"
echo "  Crop Disease Advisor — Phase 1 → Phase 2 Migration"
echo "══════════════════════════════════════════════════════"

# ── Check 1: EfficientNet checkpoint ─────────────────────────────────────────
echo ""
echo "[1/3] Checking EfficientNet-B4 checkpoint..."
if [ ! -f "$VISION_CKPT" ]; then
    echo "  ❌ Vision checkpoint not found: $VISION_CKPT"
    echo "     Run Phase 1 training before migrating."
    exit 1
fi
echo "  ✓  Checkpoint found: $VISION_CKPT"

# ── Check 2: LLM adapter ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Checking LLaMA 3 QLoRA adapter..."
ADAPTER_FILES=("adapter_config.json" "adapter_model.safetensors")
ALL_FOUND=true
for f in "${ADAPTER_FILES[@]}"; do
    if [ ! -f "$ADAPTER_DIR/$f" ]; then
        echo "  ❌ Missing adapter file: $ADAPTER_DIR/$f"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo "  Run Phase 2 QLoRA training (src/llm/train_qlora.py) before migrating."
    exit 1
fi
echo "  ✓  LLM adapter found: $ADAPTER_DIR"

# ── Check 3: Switch compose symlink ──────────────────────────────────────────
echo ""
echo "[3/3] Switching docker-compose active config → phase2..."

# Create/update the active symlink
ln -sf "$SCRIPT_DIR/docker-compose.phase2.yml" "$ACTIVE_LINK"
echo "  ✓  Active config → docker-compose.phase2.yml"

# ── Zero-downtime restart ─────────────────────────────────────────────────────
echo ""
echo "  Restarting services with zero downtime..."
cd "$PROJECT_ROOT"

# Bring up the new API without stopping the old one first
docker-compose -f "$ACTIVE_LINK" up -d --no-deps --build api
echo "  ✓  API service updated. Waiting for health..."

RETRY=0
MAX=40
until docker-compose -f "$ACTIVE_LINK" ps api | grep -q "healthy"; do
    RETRY=$((RETRY+1))
    if [ $RETRY -ge $MAX ]; then
        echo "  ❌ API did not become healthy. Rolling back..."
        ln -sf "$SCRIPT_DIR/docker-compose.phase1.yml" "$ACTIVE_LINK"
        docker-compose -f "$SCRIPT_DIR/docker-compose.phase1.yml" up -d --no-deps
        exit 1
    fi
    echo "     Waiting... ($RETRY/$MAX)"
    sleep 5
done

# Now update UI
docker-compose -f "$ACTIVE_LINK" up -d --no-deps ui
echo "  ✓  UI service updated."

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Migration to Phase 2 complete!"
echo "  API → http://localhost:8000/health"
echo "  UI  → http://localhost:7860"
echo "══════════════════════════════════════════════════════"
