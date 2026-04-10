#!/bin/bash
set -e

echo "⚙  Starting FastAPI Phase 2 server..."
uvicorn src.api.main_phase2:app --host 0.0.0.0 --port 8000 --log-level "${LOG_LEVEL:-info}" &

echo "⏳ Waiting for API to be healthy..."
MAX_RETRIES=60
RETRY=0
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    RETRY=$((RETRY+1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "❌ API did not start after ${MAX_RETRIES} retries."
        exit 1
    fi
    echo "   Attempt ${RETRY}/${MAX_RETRIES} — retrying in 3s..."
    sleep 3
done

echo "✓  API is healthy! Starting Phase 2 Gradio UI..."
python3 src/ui/gradio_phase2.py
