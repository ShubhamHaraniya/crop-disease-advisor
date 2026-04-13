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

# ── Build Vite frontend (Dependencies) ─────────────────────────────────────────
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci --silent

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Build Vite frontend (Production Build) ─────────────────────────────────────
RUN cd frontend && npm run build
# Output: frontend/dist/

# ── Environment: CPU-only Phase 1 (DISEASE_DB treatment plans, no LLM) ────────
ENV PHASE=1
ENV PYTHONUNBUFFERED=1

# ── Non-root user ──────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Launch Application ────────────────────────────────────────────────────────
CMD ["python", "app.py"]
