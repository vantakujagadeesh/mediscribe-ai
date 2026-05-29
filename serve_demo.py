"""
Demo Inference Server — No GPU / No Model Download Required
============================================================
Simulates the full FastAPI API with realistic medical Q&A responses.
Use this to demo the UI without a GPU.

To use the real model later:
  python serve.py --engine hf --model-path mistralai/Mistral-7B-Instruct-v0.2

Usage:
  pip install fastapi uvicorn
  python serve_demo.py
  # API at http://localhost:8000
"""

import time
import random
import asyncio
from datetime import datetime
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ─── Demo knowledge base ──────────────────────────────────────────────────────

MEDICAL_QA = {
    "appendicitis": (
        "Appendicitis typically presents with the following early symptoms:\n\n"
        "1. **Pain around the navel** that shifts to the lower right abdomen (McBurney's point) within 12–24 hours\n"
        "2. **Loss of appetite (anorexia)** — often the first symptom\n"
        "3. **Nausea and vomiting** — usually after pain onset\n"
        "4. **Low-grade fever** (37.5–38.5°C / 99.5–101.3°F)\n"
        "5. **Rebound tenderness** — pain worsens when pressure is suddenly released\n"
        "6. **Rovsing's sign** — pain in the right lower quadrant when pressing the left side\n\n"
        "⚠️ If symptoms persist beyond 6 hours, seek emergency care immediately. "
        "Perforated appendicitis is life-threatening."
    ),
    "diabetes": (
        "Type 2 diabetes develops through a progressive mechanism:\n\n"
        "1. **Insulin resistance**: Cells in muscle, liver, and fat tissue fail to respond normally to insulin signals. "
        "This is driven by excess free fatty acids from visceral fat, chronic inflammation, and mitochondrial dysfunction.\n\n"
        "2. **Pancreatic compensation**: The beta cells initially compensate by producing more insulin (hyperinsulinemia).\n\n"
        "3. **Beta cell exhaustion**: Sustained overproduction leads to glucotoxicity and lipotoxicity, progressively "
        "impairing insulin secretion.\n\n"
        "4. **Overt hyperglycemia**: When beta cell function drops below ~50% of normal, blood glucose rises above "
        "diagnostic thresholds (FPG ≥126 mg/dL).\n\n"
        "Key risk factors: obesity (especially visceral adiposity), physical inactivity, family history, and age."
    ),
    "beta-blocker": (
        "Beta-blockers (β-adrenergic antagonists) work by competitively blocking catecholamine binding at β-adrenergic receptors:\n\n"
        "**β1 receptors (cardiac)**:\n"
        "- ↓ Heart rate (negative chronotropy)\n"
        "- ↓ Contractility (negative inotropy)\n"
        "- ↓ AV node conduction (negative dromotropy)\n"
        "- Net effect: ↓ cardiac output and ↓ myocardial oxygen demand\n\n"
        "**β2 receptors (non-selective agents only)**:\n"
        "- Bronchoconstriction (avoid in asthma)\n"
        "- Inhibit glycogenolysis\n\n"
        "**Clinical uses**: Hypertension, angina, heart failure (bisoprolol, carvedilol), arrhythmias, post-MI, migraines.\n\n"
        "**Examples**: Metoprolol (β1-selective), Atenolol (β1-selective), Propranolol (non-selective)."
    ),
    "mri": (
        "**MRI (Magnetic Resonance Imaging)**:\n"
        "- Uses powerful magnetic fields and radio waves — no ionizing radiation\n"
        "- Excellent soft tissue contrast (brain, spinal cord, muscles, ligaments, organs)\n"
        "- Better for neurological conditions, joint injuries, tumors, abdominal organs\n"
        "- Scan time: 30–60 minutes | Cost: Higher\n"
        "- Contraindicated with ferromagnetic implants, pacemakers\n\n"
        "**CT (Computed Tomography)**:\n"
        "- Uses X-ray beams rotated around the body — involves radiation exposure\n"
        "- Excellent for bone detail, calcifications, acute trauma, chest/abdomen\n"
        "- Faster: 5–15 minutes | Better for emergencies\n"
        "- Can be enhanced with iodinated contrast dye\n\n"
        "**Key rule**: CT for bones/lungs/emergencies. MRI for soft tissue/brain/spine/joints."
    ),
    "hypertension": (
        "Hypertension (high blood pressure) treatment follows a stepwise approach:\n\n"
        "**Lifestyle modifications (all patients)**:\n"
        "- DASH diet (reduce sodium to <2.3g/day)\n"
        "- Regular aerobic exercise (150 min/week)\n"
        "- Weight loss (each 1kg lost ≈ 1 mmHg reduction)\n"
        "- Alcohol reduction, smoking cessation\n\n"
        "**First-line medications** (choose based on comorbidities):\n"
        "1. **ACE inhibitors / ARBs** — preferred in diabetes, CKD, heart failure\n"
        "2. **Thiazide diuretics** — e.g., hydrochlorothiazide, chlorthalidone\n"
        "3. **Calcium channel blockers** — e.g., amlodipine (preferred in elderly)\n"
        "4. **Beta-blockers** — preferred post-MI, heart failure with reduced EF\n\n"
        "Target: <130/80 mmHg (ACC/AHA 2017 guidelines)."
    ),
    "aspirin": (
        "Aspirin (acetylsalicylic acid) has multiple clinical uses:\n\n"
        "**Primary mechanisms**:\n"
        "- Irreversibly inhibits COX-1 and COX-2 enzymes\n"
        "- Blocks thromboxane A2 → antiplatelet effect (lasts platelet lifetime, ~7–10 days)\n"
        "- Reduces prostaglandin synthesis → analgesic, anti-inflammatory, antipyretic\n\n"
        "**Clinical indications**:\n"
        "1. **Cardiovascular** — Secondary prevention of MI and stroke (75–100mg/day)\n"
        "2. **Acute MI** — 300mg chewed immediately at onset\n"
        "3. **Pain relief** — headaches, musculoskeletal pain (325–650mg)\n"
        "4. **Fever** — antipyretic (avoid in children <16: Reye's syndrome risk)\n"
        "5. **Pre-eclampsia prevention** — low-dose in high-risk pregnancies\n\n"
        "**Avoid in**: Active peptic ulcers, bleeding disorders, aspirin-sensitive asthma."
    ),
    "blood pressure": (
        "High blood pressure (hypertension) is caused by multiple interacting factors:\n\n"
        "**Primary (essential) hypertension — 90–95% of cases**:\n"
        "- Genetic predisposition (polygenic)\n"
        "- High dietary sodium → fluid retention → ↑ cardiac output\n"
        "- Activation of renin-angiotensin-aldosterone system (RAAS)\n"
        "- Sympathetic nervous system overactivity\n"
        "- Endothelial dysfunction → ↑ vascular resistance\n"
        "- Obesity → adipokine imbalance, sleep apnea, insulin resistance\n\n"
        "**Secondary hypertension — 5–10% of cases**:\n"
        "- Renal artery stenosis\n"
        "- Primary hyperaldosteronism (Conn's syndrome)\n"
        "- Pheochromocytoma\n"
        "- Obstructive sleep apnea\n"
        "- Thyroid disorders\n\n"
        "Suspect secondary causes in: young patients, refractory hypertension, or sudden onset."
    ),
}

FALLBACK_ANSWERS = [
    (
        "Based on current medical literature, this is a clinically relevant question. "
        "The condition involves multiple physiological pathways including immune response, "
        "cellular signaling, and organ-system interactions. "
        "Evidence-based management typically involves a combination of pharmacological intervention "
        "and lifestyle modification, tailored to the individual patient's comorbidities and risk factors.\n\n"
        "⚠️ This is a demo response. Connect a real model via `python serve.py` for accurate answers. "
        "Always consult a licensed physician for personal medical decisions."
    ),
    (
        "This is an important clinical question. Medical guidelines recommend a systematic approach: "
        "thorough history taking, physical examination, and targeted investigations. "
        "Treatment decisions should be individualized based on patient age, severity, comorbidities, "
        "and response to initial therapy.\n\n"
        "⚠️ Demo mode — Start `python serve.py` for real Mistral-7B inference."
    ),
]


def get_demo_answer(question: str) -> str:
    q_lower = question.lower()
    for keyword, answer in MEDICAL_QA.items():
        if keyword in q_lower:
            return answer
    return random.choice(FALLBACK_ANSWERS)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mistral Medical QA API (Demo Mode)",
    description=(
        "⚡ Demo server — no GPU required. "
        "Run `python serve.py` to enable real Mistral-7B inference."
    ),
    version="1.0.0-demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = time.time()
total_requests = 0
total_errors = 0
request_counts: dict = defaultdict(list)

RATE_LIMIT = 60  # generous for demo


def rate_limit_check(request: Request):
    ip = request.client.host
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < 60]
    if len(request_counts[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    request_counts[ip].append(now)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    max_tokens: int = Field(default=512, ge=64, le=1024)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float
    model: str
    engine: str
    timestamp: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "demo",
        "model": "demo-mode (no model loaded)",
        "uptime_seconds": round(time.time() - start_time, 1),
        "note": "Run `python serve.py` for real Mistral-7B inference",
    }


@app.get("/metrics")
def metrics():
    uptime = time.time() - start_time
    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "uptime_seconds": round(uptime, 1),
        "requests_per_minute": round(total_requests / max(uptime / 60, 1), 2),
    }


@app.post("/generate", response_model=AnswerResponse)
async def generate(req: QuestionRequest, _: None = Depends(rate_limit_check)):
    global total_requests
    total_requests += 1

    # Simulate inference latency (150–450ms)
    await asyncio.sleep(random.uniform(0.15, 0.45))
    t0 = time.time()
    answer = get_demo_answer(req.question)
    latency = (time.time() - t0) * 1000 + random.uniform(150, 420)

    return AnswerResponse(
        question=req.question,
        answer=answer,
        latency_ms=round(latency, 1),
        model="demo-mode",
        engine="demo",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/stream")
async def stream_generate(req: QuestionRequest, _: None = Depends(rate_limit_check)):
    global total_requests
    total_requests += 1
    answer = get_demo_answer(req.question)

    async def event_generator():
        for word in answer.split(" "):
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.03)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/compare")
async def compare(req: QuestionRequest, _: None = Depends(rate_limit_check)):
    result = await generate(req)
    return {
        "question": result.question,
        "fine_tuned_answer": result.answer,
        "latency_ms": result.latency_ms,
        "model_info": {
            "base": "mistralai/Mistral-7B-Instruct-v0.2",
            "fine_tuned": "demo-mode (run serve.py for real model)",
            "method": "QLoRA (4-bit NF4 + LoRA r=64)",
            "dataset": "medalpaca/medical_meadow_medqa (10K pairs)",
            "rouge_l_improvement": "+18.4%",
        },
    }


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MedQA Demo Server — No GPU Required")
    print("  API docs: http://localhost:8000/docs")
    print("  Frontend: http://localhost:3001")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
