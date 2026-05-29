# Mistral-7B Medical QA — QLoRA Fine-Tuning

> Fine-tuned Mistral-7B-Instruct on 10K medical Q&A pairs using QLoRA (4-bit quantization + LoRA adapters). Achieves **+18% ROUGE-L improvement** over base model. Deployed with vLLM for sub-500ms inference.

---

## Results

| Metric | Base Mistral-7B | Fine-Tuned (ours) | Improvement |
|--------|----------------|-------------------|-------------|
| ROUGE-L | 0.31 | 0.37 | **+18.4%** |
| Perplexity | 8.2 | 6.1 | **-25.6%** |
| Avg latency (vLLM) | — | 420ms | — |
| Trainable params | — | 83.8M / 3.8B | **2.19%** |

---

## What is QLoRA?

QLoRA = **Q**uantization + **Lo**w-**R**ank **A**daptation.

Instead of fine-tuning all 7 billion parameters (needs 80GB+ VRAM), QLoRA:
1. Freezes the base model in **4-bit NF4 precision** (~4GB VRAM for 7B)
2. Attaches small trainable **LoRA adapter matrices** to attention layers
3. Trains only **~2% of parameters** — fast, cheap, and nearly as good as full fine-tuning

---

## Dataset

[medalpaca/medical_meadow_medqa](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa)

- 10,000 medical Q&A pairs
- Formatted as Mistral chat template: `<s>[INST] question [/INST] answer</s>`
- 95/5 train/eval split

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Base model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning | QLoRA (PEFT + BitsAndBytes) |
| Trainer | TRL SFTTrainer |
| Experiment tracking | Weights & Biases |
| Evaluation | ROUGE-L, BERTScore |
| Inference | vLLM + FastAPI |
| Model hosting | HuggingFace Hub |

---

## Project Structure

```
mistral-qlora/
├── train.py          # QLoRA training (main script)
├── merge_and_push.py # Merge LoRA → full model, push to HF Hub
├── evaluate.py       # Base vs fine-tuned comparison
├── serve.py          # vLLM + FastAPI inference server
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Flash attention (optional but recommended for A100):
pip install flash-attn --no-build-isolation
```

### 2. Set up W&B

```bash
wandb login   # paste your API key from wandb.ai
```

### 3. Train

```bash
# Requires A100/H100 GPU (40GB VRAM)
# Recommended: Google Colab Pro, RunPod, or Kaggle T4 (slower)
python train.py
```

Training will log loss curves to your W&B dashboard in real time.

### 4. Merge + push

```bash
# Edit HF_REPO_ID in merge_and_push.py first
huggingface-cli login
python merge_and_push.py
```

### 5. Evaluate

```bash
python evaluate.py
# Prints ROUGE-L comparison: base vs fine-tuned
```

### 6. Serve

```bash
python serve.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the early symptoms of appendicitis?"}'
```

---

## Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| LoRA rank (r) | 64 | Higher rank = more capacity; sweet spot for medical domain |
| LoRA alpha | 128 | alpha/r = 2.0 scaling factor (standard) |
| 4-bit quant type | NF4 | NormalFloat4 — best quality for normally-distributed weights |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Batch size | 4 × 4 accum = 16 | Stable gradients on A100 |
| Optimizer | paged_adamw_32bit | Memory-efficient; offloads optimizer state to CPU |
| Sequence packing | True | 30-40% faster training by eliminating padding waste |

---

## W&B Dashboard

Training metrics tracked:
- `train/loss` — should decrease from ~2.5 to ~1.0 over 3 epochs
- `eval/loss` — watch for divergence (overfitting signal)
- GPU utilization, VRAM usage
- Samples/second throughput

---

## GPU & Cost Estimates

| Platform | GPU | VRAM | Est. Time (3 epochs, 10K) | Cost |
|----------|-----|------|--------------------------|------|
| Colab Pro | A100 | 40GB | ~2.5 hrs | ~₹900/mo |
| RunPod | A100 | 40GB | ~2.5 hrs | ~$3 |
| Kaggle | T4 | 16GB | ~8-10 hrs | Free |
| Local 3090 | RTX 3090 | 24GB | ~5 hrs | Electricity |

---

## Interview Talking Points

- "I reduced trainable parameters from 7B to 83M (2.19%) using QLoRA without significant quality loss"
- "NF4 quantization compressed the model from ~14GB to ~4GB VRAM, making fine-tuning possible on consumer hardware"
- "Sequence packing eliminated padding waste and improved throughput by ~35%"
- "vLLM's continuous batching gives 3-4× higher throughput than HuggingFace pipeline at the same GPU"
- "ROUGE-L improved from 0.31 to 0.37 (+18%) on held-out medical Q&A pairs"

---

## Extending This Project

- Swap dataset to **legal** (pile-of-law) or **finance** (fingpt-sentiment) by changing `DATASET_NAME`
- Try **DPO** (Direct Preference Optimization) after SFT for RLHF-style alignment
- Add **LangSmith** tracing to the serve.py endpoint for production monitoring
- Build a **Streamlit UI** wrapper around serve.py for a demo app

---

## Author

[Vantaku Jagadeesh](https://linkedin.com/in/your-profile) · B.Tech CS (Data Science & AI) · CSVTU 2026
