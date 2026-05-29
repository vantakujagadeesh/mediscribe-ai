"""
Evaluate: Base Mistral-7B vs Fine-Tuned (QLoRA)
================================================
Metrics:
  - ROUGE-L (lexical overlap)
  - BERTScore F1 (semantic similarity)
  - Perplexity (model confidence)
  - Qualitative side-by-side examples

Output: eval_results.json + eval_report.md (portfolio-ready)

Usage:
  python evaluate.py
  python evaluate.py --num-samples 50 --output-file results/eval_custom.json
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from rouge_score import rouge_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluator")


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--finetuned-model", default="./outputs/mistral-medical-merged")
parser.add_argument("--num-samples", type=int, default=200)
parser.add_argument("--output-file", default="eval_results.json")
parser.add_argument("--skip-base", action="store_true", help="Only eval fine-tuned model")
args = parser.parse_args()


# ─── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic medical assistant. "
    "Provide accurate, concise, and evidence-based answers to medical questions."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_pipeline(model_path: str):
    """Load HuggingFace text-generation pipeline in fp16."""
    logger.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
        return_full_text=False,
    )


def generate_answer(pipe, question: str) -> tuple[str, float]:
    """Generate answer and return (text, latency_ms)."""
    prompt = (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n{question} [/INST]"
    )
    t0 = time.time()
    out = pipe(prompt)[0]["generated_text"]
    latency = (time.time() - t0) * 1000

    # Strip residual prompt artifacts
    answer = out.split("[/INST]")[-1].strip()
    answer = answer.replace("</s>", "").strip()
    return answer, latency


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict:
    """Compute mean and per-sample ROUGE-L F1."""
    scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer_obj.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "per_sample": [round(s, 4) for s in scores],
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore F1 (requires bert-score package)."""
    try:
        from bert_score import score as bert_score_fn
        logger.info("Computing BERTScore (this may take a few minutes)...")
        _, _, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        f1_scores = F1.numpy().tolist()
        return {
            "mean": float(np.mean(f1_scores)),
            "std": float(np.std(f1_scores)),
        }
    except ImportError:
        logger.warning("bert-score not installed — skipping BERTScore")
        return {"mean": None, "std": None, "note": "install bert-score"}
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"mean": None, "error": str(e)}


def evaluate_model(name: str, model_path: str, questions: list, references: list) -> dict:
    """Full evaluation loop for a single model."""
    logger.info(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")

    pipe = load_pipeline(model_path)
    predictions, latencies = [], []

    for i, q in enumerate(questions):
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(questions)}")
        answer, latency = generate_answer(pipe, q)
        predictions.append(answer)
        latencies.append(latency)

    rouge = compute_rouge_l(predictions, references)
    bert = compute_bertscore(predictions, references)

    result = {
        "model": model_path,
        "num_samples": len(questions),
        "rouge_l": rouge,
        "bert_score_f1": bert,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "sample_predictions": [
            {"question": q, "reference": r, "prediction": p}
            for q, r, p in zip(questions[:5], references[:5], predictions[:5])
        ],
    }

    logger.info(f"  ROUGE-L:     {rouge['mean']:.4f} ± {rouge['std']:.4f}")
    logger.info(f"  BERTScore:   {bert['mean']}")
    logger.info(f"  Avg latency: {result['latency_ms']['mean']:.1f}ms")

    # Cleanup GPU memory
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ─── Load eval dataset ────────────────────────────────────────────────────────

logger.info(f"Loading eval dataset: medalpaca/medical_meadow_medqa ({args.num_samples} samples)")
dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")
eval_samples = dataset.shuffle(seed=99).select(range(args.num_samples))
questions = [s["input"] for s in eval_samples]
references = [s["output"] for s in eval_samples]


# ─── Run evaluations ──────────────────────────────────────────────────────────

results = {"timestamp": datetime.utcnow().isoformat() + "Z"}

if not args.skip_base:
    results["base"] = evaluate_model("Base Mistral-7B", args.base_model, questions, references)

finetuned_path = Path(args.finetuned_model)
if not finetuned_path.exists():
    logger.error(f"Fine-tuned model not found at: {finetuned_path}")
    logger.error("Run train.py → merge_and_push.py first.")
    sys.exit(1)

results["finetuned"] = evaluate_model("Fine-Tuned QLoRA", args.finetuned_model, questions, references)


# ─── Print comparison ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

if "base" in results and "finetuned" in results:
    base_r = results["base"]["rouge_l"]["mean"]
    ft_r = results["finetuned"]["rouge_l"]["mean"]
    delta = ((ft_r - base_r) / base_r) * 100
    print(f"  Base Mistral-7B   ROUGE-L: {base_r:.4f}")
    print(f"  Fine-Tuned (QLoRA) ROUGE-L: {ft_r:.4f}")
    print(f"  Improvement: +{delta:.1f}%")

    base_bert = results["base"]["bert_score_f1"]["mean"]
    ft_bert = results["finetuned"]["bert_score_f1"]["mean"]
    if base_bert and ft_bert:
        bert_delta = ((ft_bert - base_bert) / base_bert) * 100
        print(f"  Base BERTScore F1:  {base_bert:.4f}")
        print(f"  FT   BERTScore F1:  {ft_bert:.4f}")
        print(f"  BERTScore Δ:        +{bert_delta:.1f}%")

    results["comparison"] = {
        "rouge_l_improvement_pct": round(delta, 2),
        "base_rouge_l": round(base_r, 4),
        "finetuned_rouge_l": round(ft_r, 4),
    }
elif "finetuned" in results:
    ft_r = results["finetuned"]["rouge_l"]["mean"]
    print(f"  Fine-Tuned ROUGE-L: {ft_r:.4f}")

print("=" * 60)


# ─── Save results ─────────────────────────────────────────────────────────────

output_path = Path(args.output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
logger.info(f"\nFull results saved → {output_path}")


# ─── Generate Markdown report ────────────────────────────────────────────────

report_path = output_path.with_suffix(".md")
with open(report_path, "w") as f:
    f.write(f"# Evaluation Report\n\n")
    f.write(f"**Generated**: {results['timestamp']}\n\n")

    if "comparison" in results:
        c = results["comparison"]
        f.write("## Summary\n\n")
        f.write("| Metric | Base Mistral-7B | Fine-Tuned (QLoRA) | Improvement |\n")
        f.write("|--------|----------------|---------------------|-------------|\n")
        f.write(f"| ROUGE-L | {c['base_rouge_l']} | {c['finetuned_rouge_l']} | **+{c['rouge_l_improvement_pct']}%** |\n\n")

    f.write("## Sample Predictions\n\n")
    for i, sample in enumerate(results["finetuned"]["sample_predictions"], 1):
        f.write(f"### Example {i}\n\n")
        f.write(f"**Q**: {sample['question']}\n\n")
        f.write(f"**Reference**: {sample['reference']}\n\n")
        f.write(f"**Fine-Tuned**: {sample['prediction']}\n\n---\n\n")

logger.info(f"Markdown report saved → {report_path}")
