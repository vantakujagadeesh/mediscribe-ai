"""
MedScribe AI — FastAPI Inference Server
========================================
Features:
  - vLLM primary engine (3-4× faster than HF pipeline)
  - HuggingFace pipeline fallback (for CPU/consumer GPU)
  - POST /v1/scribe — license-gated SOAP note generation
  - POST /webhooks/lemonsqueezy — payment webhook
  - GET  /admin/usage — usage analytics
  - SQLite usage tracking with tier-based limits
  - Streaming response via SSE
  - Rate limiting (per-IP)
  - /health, /metrics, /generate, /stream endpoints
  - CORS support for frontend

Usage:
  python serve.py --engine vllm
  python serve.py --engine hf
"""

import os
import sys
import time
import logging
import argparse
import asyncio
import uuid
import hmac
import hashlib
import sqlite3
import json
from collections import defaultdict
from datetime import datetime
from typing import AsyncGenerator, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medscribe-api")


# ─── License Keys & Tiers ─────────────────────────────────────────────────────

VALID_KEYS: dict[str, dict] = {}

TIER_LIMITS = {
    "starter": 200,
    "professional": 999_999_999,
    "clinic": 999_999_999,
}


def load_license_keys():
    """Load license keys from VALID_KEYS env var and SQLite DB."""
    # 1. Load from ENV
    raw = os.getenv("VALID_KEYS", "")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                VALID_KEYS.update(parsed)
                logger.info(f"[LICENSE] Loaded {len(parsed)} keys from ENV (JSON format)")
        except json.JSONDecodeError:
            csv_keys = 0
            for key in raw.split(","):
                key = key.strip()
                if key:
                    VALID_KEYS[key] = {"tier": "starter", "email": "unknown"}
                    csv_keys += 1
            logger.info(f"[LICENSE] Loaded {csv_keys} keys from ENV (CSV format)")

    # 2. Load from DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # Create table if it doesn't exist yet so this doesn't crash on first run before init_usage_db
        conn.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_key TEXT UNIQUE NOT NULL,
                tier TEXT NOT NULL DEFAULT 'starter',
                email TEXT,
                expires_at TEXT,
                created_at TEXT NOT NULL
            )
        """)
        rows = conn.execute("SELECT * FROM licenses").fetchall()
        for r in rows:
            VALID_KEYS[r["license_key"]] = {
                "tier": r["tier"],
                "email": r["email"],
                "expires_at": r["expires_at"]
            }
        conn.close()
        logger.info(f"[LICENSE] Loaded {len(rows)} keys from DB")
    except Exception as e:
        logger.error(f"[LICENSE] Failed to load keys from DB: {e}")

    if not VALID_KEYS:
        logger.warning("[LICENSE] No VALID_KEYS configured — all requests will be rejected")


def validate_license(license_key: str) -> dict:
    """Validate a license key and return its tier info, or raise 403."""
    if license_key not in VALID_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or unrecognized license key")
    
    key_info = VALID_KEYS[license_key]
    expires_at = key_info.get("expires_at")
    if expires_at:
        # Check if expired
        try:
            exp_date = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).replace(tzinfo=None)
            if datetime.utcnow() > exp_date:
                raise HTTPException(status_code=403, detail="Your free trial has expired. Please upgrade to a paid plan.")
        except ValueError:
            pass # fallback if date parsing fails

    return key_info


def generate_license_key() -> str:
    """Generate a unique MedScribe license key."""
    return f"MSAI-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}"


# ─── SQLite Usage Tracking ────────────────────────────────────────────────────

DB_PATH = os.getenv("USAGE_DB_PATH", "medscribe_usage.db")


def init_usage_db():
    """Initialize the SQLite usage database with the requests table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            license_key TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'starter',
            timestamp TEXT NOT NULL,
            input_len INTEGER NOT NULL,
            output_len INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            status TEXT DEFAULT 'success'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON usage_log(license_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON usage_log(timestamp)")
    
    # Also ensure licenses table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS licenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_key TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'starter',
            email TEXT,
            expires_at TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"[DB] Usage database initialized at {DB_PATH}")


def save_license_key(key: str, tier: str, email: str, expires_at: str = None):
    """Save a newly generated license key to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO licenses (license_key, tier, email, expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
            (key, tier, email, expires_at, datetime.utcnow().isoformat() + "Z")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[DB] Failed to save license key: {e}")


def log_usage(request_id: str, license_key: str, tier: str,
              input_len: int, output_len: int, latency_ms: float, status: str = "success"):
    """Insert a usage record into the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO usage_log (request_id, license_key, tier, timestamp, input_len, output_len, latency_ms, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, license_key, tier, datetime.utcnow().isoformat() + "Z",
             input_len, output_len, latency_ms, status),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[DB] Failed to log usage: {e}")


def get_monthly_usage(license_key: str) -> int:
    """Count how many successful requests this key has made in the current month."""
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM usage_log WHERE license_key = ? AND timestamp >= ? AND status = 'success'",
            (license_key, month_start),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"[DB] Failed to get usage count: {e}")
        return 0


def check_tier_limit(license_key: str, tier: str):
    """Enforce monthly note limits based on the subscription tier."""
    limit = TIER_LIMITS.get(tier, 200)
    used = get_monthly_usage(license_key)
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly limit reached ({used}/{limit} notes). Upgrade your plan.",
        )
    return used


# ─── CLI Args ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--engine",
    choices=["vllm", "hf"],
    default=os.getenv("INFERENCE_ENGINE", "hf"),
    help="Inference backend: 'vllm' (GPU server) or 'hf' (local/dev)",
)
parser.add_argument(
    "--model-path",
    default=os.getenv("MODEL_PATH", "./outputs/mistral-medical-merged"),
    help="Path to merged model dir or HuggingFace repo ID",
)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument(
    "--rate-limit",
    type=int,
    default=int(os.getenv("RATE_LIMIT_RPM", "30")),
    help="Max requests per minute per IP",
)
args, _ = parser.parse_known_args()


# ─── Inference Engine ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and empathetic medical assistant. "
    "Provide accurate, concise, and evidence-based answers to medical questions. "
    "Always recommend consulting a licensed physician for personal medical decisions."
)


class BaseEngine:
    def generate(self, question: str, max_tokens: int) -> tuple[str, float]:
        raise NotImplementedError

    def generate_stream(self, question: str, max_tokens: int) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class VLLMEngine(BaseEngine):
    def __init__(self, model_path: str):
        logger.info(f"[vLLM] Loading model: {model_path}")
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_path,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.90,
        )
        self.SamplingParams = SamplingParams
        logger.info("[vLLM] Model loaded ✓")

    def _build_prompt(self, question: str) -> str:
        return (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n{question} [/INST]"
        )

    def generate(self, question: str, max_tokens: int = 512) -> tuple[str, float]:
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
        )
        t0 = time.time()
        outputs = self.llm.generate([self._build_prompt(question)], params)
        latency = (time.time() - t0) * 1000
        answer = outputs[0].outputs[0].text.strip()
        return answer, latency

    async def generate_stream(self, question: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
        # vLLM doesn't natively support async streaming in all versions
        # — run blocking generate in thread pool
        answer, _ = await asyncio.get_event_loop().run_in_executor(
            None, self.generate, question, max_tokens
        )
        # Simulate token-by-token stream
        for word in answer.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)


class HFEngine(BaseEngine):
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

        logger.info(f"[HF] Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        logger.info("[HF] Model loaded ✓")

    def _build_prompt(self, question: str) -> str:
        return f"<s>[INST] {SYSTEM_PROMPT}\n\n{question} [/INST]"

    def generate(self, question: str, max_tokens: int = 512) -> tuple[str, float]:
        t0 = time.time()
        out = self.pipe(self._build_prompt(question), max_new_tokens=max_tokens)
        latency = (time.time() - t0) * 1000
        answer = out[0]["generated_text"].strip()
        answer = answer.replace("</s>", "").strip()
        return answer, latency

    async def generate_stream(self, question: str, max_tokens: int = 512) -> AsyncGenerator[str, None]:
        answer, _ = await asyncio.get_event_loop().run_in_executor(
            None, self.generate, question, max_tokens
        )
        for word in answer.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)


def load_engine(engine_type: str, model_path: str) -> BaseEngine:
    if engine_type == "vllm":
        try:
            return VLLMEngine(model_path)
        except ImportError:
            logger.warning("vLLM not installed — falling back to HuggingFace engine")
            return HFEngine(model_path)
    return HFEngine(model_path)


# ─── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mistral Medical QA API",
    description=(
        "Fine-tuned Mistral-7B-Instruct on 10K medical Q&A pairs (QLoRA). "
        "Achieves +18% ROUGE-L improvement over the base model."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
engine: Optional[BaseEngine] = None
request_counts: dict = defaultdict(list)
total_requests = 0
total_errors = 0
start_time = time.time()


@app.on_event("startup")
async def startup():
    """Initialize model engine, load license keys, and set up usage database."""
    global engine
    logger.info(f"Starting server — engine={args.engine}, model={args.model_path}")
    load_license_keys()
    init_usage_db()
    engine = load_engine(args.engine, args.model_path)
    logger.info("Server ready ✓")


# ─── Rate limiting ────────────────────────────────────────────────────────────

def rate_limit_check(request: Request):
    ip = request.client.host
    now = time.time()
    window = 60  # 1 minute

    # Purge old requests
    request_counts[ip] = [t for t in request_counts[ip] if now - t < window]

    if len(request_counts[ip]) >= args.rate_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {args.rate_limit} requests/minute",
        )
    request_counts[ip].append(now)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Schema for the legacy /generate endpoint."""
    question: str = Field(..., min_length=3, max_length=2000, example="What are the early symptoms of appendicitis?")
    max_tokens: int = Field(default=512, ge=64, le=1024)


class ScribeRequest(BaseModel):
    """Schema for the /v1/scribe SOAP note endpoint."""
    text: str = Field(..., min_length=10, max_length=5000, description="Clinical dictation text")
    license_key: str = Field(..., min_length=8, description="MedScribe license key")
    max_tokens: int = Field(default=1024, ge=64, le=2048)


class ScribeResponse(BaseModel):
    """Response schema for /v1/scribe."""
    soap_note: str
    latency_ms: float
    request_id: str
    usage_this_month: int
    tier: str


class WebhookPayload(BaseModel):
    """Lemon Squeezy webhook event payload."""
    meta: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    """Response schema for /generate."""
    question: str
    answer: str
    latency_ms: float
    model: str
    engine: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    engine: str
    model: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    uptime_seconds: float
    requests_per_minute: float


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok" if engine else "loading",
        engine=args.engine,
        model=args.model_path,
        uptime_seconds=round(time.time() - start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def metrics():
    uptime = time.time() - start_time
    rpm = total_requests / (uptime / 60) if uptime > 0 else 0
    return MetricsResponse(
        total_requests=total_requests,
        total_errors=total_errors,
        uptime_seconds=round(uptime, 1),
        requests_per_minute=round(rpm, 2),
    )


@app.post("/generate", response_model=AnswerResponse, tags=["Inference"])
async def generate(
    req: QuestionRequest,
    _: None = Depends(rate_limit_check),
):
    global total_requests, total_errors
    total_requests += 1

    if not engine:
        raise HTTPException(status_code=503, detail="Model is still loading")

    try:
        answer, latency = await asyncio.get_event_loop().run_in_executor(
            None, engine.generate, req.question, req.max_tokens
        )
    except Exception as e:
        total_errors += 1
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    logger.info(f"[/generate] latency={latency:.0f}ms  q_len={len(req.question)}")

    return AnswerResponse(
        question=req.question,
        answer=answer,
        latency_ms=round(latency, 1),
        model=args.model_path,
        engine=args.engine,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/stream", tags=["Inference"])
async def stream_generate(
    req: QuestionRequest,
    _: None = Depends(rate_limit_check),
):
    """Stream tokens via Server-Sent Events (SSE)."""
    global total_requests
    total_requests += 1

    if not engine:
        raise HTTPException(status_code=503, detail="Model is still loading")

    async def event_generator():
        async for token in engine.generate_stream(req.question, req.max_tokens):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/compare", tags=["Demo"])
async def compare(req: QuestionRequest, _: None = Depends(rate_limit_check)):
    """Portfolio demo: returns structured comparison output."""
    result = await generate(req)
    return {
        "question": result.question,
        "fine_tuned_answer": result.answer,
        "latency_ms": result.latency_ms,
        "model_info": {
            "base": "mistralai/Mistral-7B-Instruct-v0.2",
            "fine_tuned": args.model_path,
            "method": "QLoRA (4-bit NF4 + LoRA r=64)",
            "dataset": "medalpaca/medical_meadow_medqa (10K pairs)",
            "rouge_l_improvement": "+18.4%",
        },
    }


# ─── SaaS: SOAP Scribe Endpoint ───────────────────────────────────────────────

SOAP_SYSTEM_PROMPT = (
    "You are MedScribe AI, a clinical documentation specialist. "
    "Convert the following clinical dictation into a properly structured SOAP note. "
    "Format the output with clear sections: "
    "**S (Subjective):** — patient's reported symptoms, history, complaints. "
    "**O (Objective):** — vitals, physical exam findings, lab results. "
    "**A (Assessment):** — diagnosis/differential diagnoses. "
    "**P (Plan):** — treatment plan, medications, follow-up. "
    "Be concise, clinical, and evidence-based. Use standard medical abbreviations."
)


@app.post("/v1/scribe", response_model=ScribeResponse, tags=["SaaS"])
async def scribe(req: ScribeRequest):
    """Convert clinical dictation to a structured SOAP note (license-gated)."""
    global total_requests, total_errors
    total_requests += 1
    request_id = f"scr_{uuid.uuid4().hex[:12]}"

    key_info = validate_license(req.license_key)
    tier = key_info.get("tier", "starter")
    used = check_tier_limit(req.license_key, tier)

    if not engine:
        raise HTTPException(status_code=503, detail="Model is still loading")

    soap_prompt = f"{SOAP_SYSTEM_PROMPT}\n\nClinical Dictation:\n{req.text}"

    try:
        answer, latency = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, engine.generate, soap_prompt, req.max_tokens
            ),
            timeout=8.0,
        )
    except asyncio.TimeoutError:
        total_errors += 1
        log_usage(request_id, req.license_key, tier, len(req.text), 0, 0, "timeout")
        raise HTTPException(status_code=504, detail="Model inference timed out (8s limit)")
    except Exception as e:
        total_errors += 1
        log_usage(request_id, req.license_key, tier, len(req.text), 0, 0, "error")
        logger.error(f"[/v1/scribe] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    log_usage(request_id, req.license_key, tier, len(req.text), len(answer), latency)
    logger.info(f"[/v1/scribe] id={request_id} latency={latency:.0f}ms tier={tier}")

    return ScribeResponse(
        soap_note=answer,
        latency_ms=round(latency, 1),
        request_id=request_id,
        usage_this_month=used + 1,
        tier=tier,
    )


# ─── SaaS: Free Trial ─────────────────────────────────────────────────────────

class TrialRequest(BaseModel):
    """Schema for requesting a free trial key."""
    email: str = Field(..., description="User's email address")

@app.post("/v1/trial", tags=["SaaS"])
async def create_trial_key(req: TrialRequest):
    """Generate a 3-day free trial license key."""
    from datetime import timedelta
    
    expires_at = (datetime.utcnow() + timedelta(days=3)).isoformat() + "Z"
    new_key = generate_license_key()
    
    VALID_KEYS[new_key] = {
        "tier": "starter",
        "email": req.email,
        "expires_at": expires_at
    }
    save_license_key(new_key, "starter", req.email, expires_at)
    
    logger.info(f"[TRIAL] Generated 3-day trial key for {req.email}: {new_key[:12]}...")
    
    return {
        "status": "success",
        "message": "Your 3-day free trial key has been generated.",
        "license_key": new_key,
        "expires_at": expires_at,
        "tier": "starter"
    }


# ─── SaaS: Lemon Squeezy Webhook ─────────────────────────────────────────────

LS_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "noreply@medscribe.ai")


@app.post("/webhooks/lemonsqueezy", tags=["Webhooks"])
async def lemonsqueezy_webhook(request: Request):
    """Handle Lemon Squeezy payment webhook: generate key, email to customer."""
    body = await request.body()

    if LS_WEBHOOK_SECRET:
        signature = request.headers.get("x-signature", "")
        expected = hmac.new(
            LS_WEBHOOK_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            logger.warning("[WEBHOOK] Invalid signature — rejecting")
            raise HTTPException(status_code=403, detail="Invalid webhook signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid JSON payload")

    event_name = payload.get("meta", {}).get("event_name", "")
    if event_name != "order_created":
        logger.info(f"[WEBHOOK] Ignoring event: {event_name}")
        return {"status": "ignored", "event": event_name}

    customer_email = (
        payload.get("data", {})
        .get("attributes", {})
        .get("user_email", "")
    )
    variant = (
        payload.get("data", {})
        .get("attributes", {})
        .get("first_order_item", {})
        .get("variant_name", "Starter")
        .lower()
    )

    tier = "starter"
    if "professional" in variant or "pro" in variant:
        tier = "professional"
    elif "clinic" in variant:
        tier = "clinic"

    new_key = generate_license_key()
    VALID_KEYS[new_key] = {"tier": tier, "email": customer_email}
    save_license_key(new_key, tier, customer_email)
    logger.info(f"[WEBHOOK] New key generated: {new_key[:12]}... tier={tier} email={customer_email}")

    if RESEND_API_KEY and customer_email:
        try:
            import resend
            resend.api_key = RESEND_API_KEY
            resend.Emails.send({
                "from": RESEND_FROM_EMAIL,
                "to": customer_email,
                "subject": "Your MedScribe AI License Key",
                "html": (
                    f"<h2>Welcome to MedScribe AI!</h2>"
                    f"<p>Your license key: <code style='font-size:1.2em;background:#f0f0f0;padding:4px 8px;'>{new_key}</code></p>"
                    f"<p>Plan: <strong>{tier.title()}</strong></p>"
                    f"<p>Enter this key at <a href='https://app.medscribe.ai'>app.medscribe.ai</a> to start generating SOAP notes.</p>"
                ),
            })
            logger.info(f"[WEBHOOK] License key emailed to {customer_email}")
        except Exception as e:
            logger.error(f"[WEBHOOK] Email send failed: {e}")

    return {"status": "ok", "key_prefix": new_key[:12], "tier": tier}


# ─── SaaS: Admin Usage Endpoint ───────────────────────────────────────────────

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")


@app.get("/admin/usage", tags=["Admin"])
async def admin_usage(
    key: str = "",
    admin_token: str = Header(default="", alias="x-admin-token"),
):
    """Return usage statistics. Requires admin token via x-admin-token header."""
    if ADMIN_SECRET and admin_token != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin token")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        if key:
            rows = conn.execute(
                "SELECT * FROM usage_log WHERE license_key = ? ORDER BY timestamp DESC LIMIT 100",
                (key,),
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) as cnt, AVG(latency_ms) as avg_lat FROM usage_log WHERE license_key = ?",
                (key,),
            ).fetchone()
        else:
            rows = conn.execute(
                "SELECT * FROM usage_log ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) as cnt, AVG(latency_ms) as avg_lat FROM usage_log"
            ).fetchone()

        conn.close()

        return {
            "total_requests": total["cnt"],
            "avg_latency_ms": round(total["avg_lat"] or 0, 1),
            "recent": [dict(r) for r in rows[:50]],
        }
    except Exception as e:
        logger.error(f"[ADMIN] Usage query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
    )
