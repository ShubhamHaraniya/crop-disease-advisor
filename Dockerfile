FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (CPU-only, no LLM) ────────────────────────────────────
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# ── Build Vite frontend ────────────────────────────────────────────────────────
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci --silent

COPY frontend/ ./frontend/
RUN cd frontend && npm run build
# Output: frontend/dist/

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Environment: CPU-only Phase 1 (DISEASE_DB treatment plans, no LLM) ────────
ENV PHASE=1
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# ── HuggingFace Spaces requires non-root user ──────────────────────────────────
RUN useradd -m -u 1000 hfuser && chown -R hfuser /app
USER hfuser

# ── Expose HF port ────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Launch FastAPI (serves API + Vite static files) ───────────────────────────
CMD ["python", "-m", "uvicorn", "src.api.main_phase2:app", "--host", "0.0.0.0", "--port", "7860"]
