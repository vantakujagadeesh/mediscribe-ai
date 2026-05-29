"""
Merge LoRA adapters → full model + push to HuggingFace Hub
===========================================================
Run AFTER training completes.
Usage: python merge_and_push.py [--no-push]

Steps:
  1. Load base model in fp16 (no quant — needed for merge)
  2. Load LoRA adapter and merge weights
  3. Save merged model locally
  4. Push to HuggingFace Hub (optional)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("merge-and-push")


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--no-push", action="store_true", help="Skip HF Hub push")
parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--adapter-dir", default="./outputs/mistral-medical-qlora")
parser.add_argument("--merged-dir", default="./outputs/mistral-medical-merged")
parser.add_argument(
    "--hf-repo",
    default=os.getenv("HF_REPO_ID", "your-username/mistral-7b-medical-qlora"),
)
args = parser.parse_args()


# ─── Validate paths ───────────────────────────────────────────────────────────

adapter_path = Path(args.adapter_dir)
if not adapter_path.exists():
    logger.error(f"Adapter directory not found: {adapter_path}")
    logger.error("Run train.py first to generate the LoRA adapter weights.")
    sys.exit(1)

merged_path = Path(args.merged_dir)
merged_path.mkdir(parents=True, exist_ok=True)


# ─── 1. Load base model in fp16 ───────────────────────────────────────────────

logger.info(f"Loading base model: {args.base_model}  [fp16]")
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)


# ─── 2. Load LoRA adapter & merge ─────────────────────────────────────────────

logger.info(f"Loading LoRA adapter from: {args.adapter_dir}")
model = PeftModel.from_pretrained(base_model, args.adapter_dir)

logger.info("Merging LoRA weights into base model (merge_and_unload)...")
model = model.merge_and_unload()   # fuses LoRA Δ into frozen weights
model.eval()

logger.info(f"Post-merge parameter count: {sum(p.numel() for p in model.parameters()):,}")


# ─── 3. Save merged model ─────────────────────────────────────────────────────

logger.info(f"Saving merged model → {args.merged_dir}")
model.save_pretrained(args.merged_dir, safe_serialization=True)
tokenizer.save_pretrained(args.merged_dir)

# Verify saved files
saved_files = list(merged_path.glob("*.safetensors"))
logger.info(f"Saved {len(saved_files)} safetensor shard(s) to {args.merged_dir}")
logger.info("Merge complete!")


# ─── 4. Push to HuggingFace Hub ───────────────────────────────────────────────

if not args.no_push:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.error(
            "HuggingFace token not found. "
            "Set HF_TOKEN env var or run: huggingface-cli login"
        )
        sys.exit(1)

    logger.info(f"Pushing to HuggingFace Hub: {args.hf_repo}")
    model.push_to_hub(args.hf_repo, token=hf_token, safe_serialization=True)
    tokenizer.push_to_hub(args.hf_repo, token=hf_token)
    logger.info(f"Model live at: https://huggingface.co/{args.hf_repo}")
else:
    logger.info("Skipping HF Hub push (--no-push flag set).")
    logger.info(f"To push later: huggingface-cli upload {args.merged_dir} {args.hf_repo}")
