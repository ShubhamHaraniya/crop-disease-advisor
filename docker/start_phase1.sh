#!/bin/bash
set -e

echo "⚙  Starting FastAPI server..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level "${LOG_LEVEL:-info}" &

# Wait until the API is healthy
echo "⏳ Waiting for API to be ready..."
MAX_RETRIES=30
RETRY=0
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    RETRY=$((RETRY+1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "❌ API did not start after ${MAX_RETRIES} retries. Exiting."
        exit 1
    fi
    echo "   Attempt ${RETRY}/${MAX_RETRIES} — retrying in 2s..."
    sleep 2
done

echo "✓  API is healthy! Starting Gradio UI..."
python3 src/ui/gradio_phase1.py
