# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Crop Disease Advisor
# Common development commands. Run: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help setup setup-phase2 verify lint format test test-fast \
        preprocess train evaluate register serve-api serve-ui \
        docker-phase1 docker-phase2 migrate generate-dataset train-llm \
        upload clean

PYTHON    := python3
VENV      := ./venv
PHASE     ?= 1

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🌾 Crop Disease Advisor — Makefile Commands"
	@echo "  ════════════════════════════════════════════"
	@echo ""
	@echo "  Setup:"
	@echo "    make setup           — Install Phase 1 environment"
	@echo "    make setup-phase2    — Install Phase 2 environment"
	@echo "    make verify          — Check all packages importable"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make lint            — flake8 + black check"
	@echo "    make format          — auto-format with black"
	@echo ""
	@echo "  Testing:"
	@echo "    make test            — Run all unit tests"
	@echo "    make test-fast       — Skip slow/GPU/LLM tests"
	@echo ""
	@echo "  Phase 1 Pipeline:"
	@echo "    make preprocess      — Preprocess PlantVillage dataset"
	@echo "    make train           — Train EfficientNet-B4"
	@echo "    make evaluate        — Evaluate & generate metrics"
	@echo "    make register        — Register model to W&B + HF Hub"
	@echo ""
	@echo "  Phase 2 Pipeline:"
	@echo "    make generate-dataset — Generate 12K LLM instruction pairs"
	@echo "    make train-llm       — QLoRA fine-tune LLaMA 3"
	@echo ""
	@echo "  Serving:"
	@echo "    make serve-api       — Start FastAPI server (phase \$$PHASE)"
	@echo "    make serve-ui        — Start Gradio UI (phase \$$PHASE)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-phase1   — Build & run Phase 1 compose stack"
	@echo "    make docker-phase2   — Build & run Phase 2 compose stack"
	@echo "    make migrate         — Migrate Phase1 → Phase2 (zero-downtime)"
	@echo ""
	@echo "  SLURM:"
	@echo "    make slurm-all       — Submit all jobs with dependency chain"
	@echo "    make slurm-phase1    — Submit Phase 1 training only"
	@echo ""
	@echo "  Other:"
	@echo "    make upload          — Upload models to HuggingFace Hub"
	@echo "    make clean           — Remove __pycache__, .pytest_cache, logs"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
setup:
	@bash setup_env.sh phase1

setup-phase2:
	@bash setup_env.sh phase2

verify:
	@$(PYTHON) verify_env.py

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=120
	@echo "Running black (check)..."
	black --check src/ tests/

format:
	@echo "Auto-formatting with black..."
	black src/ tests/

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -m "not slow and not gpu and not llm"

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# ─── Phase 1 Pipeline ─────────────────────────────────────────────────────────
preprocess:
	$(PYTHON) src/vision/preprocess.py \
		--raw_dir data/raw/plantvillage \
		--output_dir data/processed \
		--aug_dir notebooks/sample_augmentations

train:
	$(PYTHON) src/vision/train.py --config configs/vision_config.yaml

evaluate:
	$(PYTHON) src/vision/evaluate.py --wandb

register:
	$(PYTHON) pipelines/register_model.py \
		--stage staging \
		--hf_username $(HF_USERNAME) \
		--hf_token    $(HF_TOKEN)

# ─── Phase 2 Pipeline ─────────────────────────────────────────────────────────
generate-dataset:
	$(PYTHON) src/llm/generate_dataset.py --n 12000 \
		--output data/processed/llm_instructions.jsonl

train-llm:
	$(PYTHON) src/llm/train_qlora.py --config configs/llm_config.yaml

# ─── Serving ──────────────────────────────────────────────────────────────────
serve-api:
ifeq ($(PHASE),2)
	uvicorn src.api.main_phase2:app --host 0.0.0.0 --port 8000 --reload
else
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
endif

serve-ui:
ifeq ($(PHASE),2)
	$(PYTHON) src/ui/gradio_phase2.py
else
	$(PYTHON) src/ui/gradio_phase1.py
endif

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-phase1:
	docker-compose -f docker/docker-compose.phase1.yml up --build

docker-phase2:
	docker-compose -f docker/docker-compose.phase2.yml up --build

migrate:
	bash docker/migrate_phase1_to_phase2.sh

# ─── SLURM ────────────────────────────────────────────────────────────────────
slurm-all:
	bash slurm/run_all.sh all

slurm-phase1:
	bash slurm/run_all.sh phase1

slurm-phase2:
	bash slurm/run_all.sh phase2

# ─── Deploy ───────────────────────────────────────────────────────────────────
upload:
	$(PYTHON) pipelines/upload_models.py \
		--username $(HF_USERNAME) \
		--token    $(HF_TOKEN)

# ─── DVC Pipeline ────────────────────────────────────────────────────────────
dvc-run:
	dvc repro

dvc-dag:
	dvc dag

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f coverage.xml
	rm -rf htmlcov/
	@echo "✓ Clean done."
