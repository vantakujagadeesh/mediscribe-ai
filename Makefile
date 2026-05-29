# Mistral-7B Medical QA — Makefile
# Usage: make help

.PHONY: help install train merge evaluate serve ui test clean docker-build docker-up

## ─── Colors ──────────────────────────────────────────────────────────────────
CYAN  = \033[0;36m
RESET = \033[0m
BOLD  = \033[1m

help:
	@echo ""
	@echo "$(BOLD)Mistral-7B Medical QA — QLoRA Pipeline$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "$(CYAN)make install$(RESET)       Install Python dependencies"
	@echo "$(CYAN)make train$(RESET)         Run QLoRA fine-tuning"
	@echo "$(CYAN)make merge$(RESET)         Merge LoRA → full model"
	@echo "$(CYAN)make merge-push$(RESET)    Merge + push to HuggingFace Hub"
	@echo "$(CYAN)make evaluate$(RESET)      Run ROUGE-L + BERTScore evaluation"
	@echo "$(CYAN)make serve$(RESET)         Start FastAPI inference server (HF mode)"
	@echo "$(CYAN)make serve-vllm$(RESET)    Start server with vLLM engine"
	@echo "$(CYAN)make ui$(RESET)            Start Next.js frontend"
	@echo "$(CYAN)make test$(RESET)          Run API tests"
	@echo "$(CYAN)make docker-up$(RESET)     Start full stack via Docker Compose"
	@echo "$(CYAN)make clean$(RESET)         Remove generated outputs"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

train:
	python train.py

merge:
	python merge_and_push.py --no-push

merge-push:
	python merge_and_push.py

evaluate:
	python evaluate.py --num-samples 200

evaluate-quick:
	python evaluate.py --num-samples 50 --skip-base

serve:
	python serve.py --engine hf --port 8000

serve-vllm:
	python serve.py --engine vllm --port 8000

test-api:
	@echo "Testing /health endpoint..."
	curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@echo "Testing /generate endpoint..."
	curl -s -X POST http://localhost:8000/generate \
		-H "Content-Type: application/json" \
		-d '{"question": "What are the early symptoms of appendicitis?"}' \
		| python -m json.tool

ui:
	cd ui && npm run dev

ui-install:
	cd ui && npm install

ui-build:
	cd ui && npm run build

test:
	pytest tests/ -v

docker-build:
	docker build -t mistral-medical-api -f docker/Dockerfile.serve .

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

clean:
	rm -rf outputs/mistral-medical-qlora
	rm -rf outputs/mistral-medical-merged
	rm -f eval_results.json eval_results.md training.log
	@echo "✓ Cleaned generated outputs"
