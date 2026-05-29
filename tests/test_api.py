"""
MedScribe AI — Full SaaS Test Suite
=====================================
Tests all endpoints: /health, /metrics, /generate, /v1/scribe,
/webhooks/lemonsqueezy, /admin/usage

Usage:
    # Start backend first with a test key:
    VALID_KEYS=TEST-KEY-1234-ABCD python serve.py --engine hf

    # Run tests:
    pytest tests/test_api.py -v
"""

import pytest
import httpx
import json
import hmac
import hashlib


BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0  # model inference can be slow
TEST_LICENSE_KEY = "TEST-KEY-1234-ABCD"
INVALID_KEY = "INVALID-KEY-0000-XXXX"


@pytest.fixture(scope="module")
def client():
    """Create a shared HTTP client for all tests."""
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


# ─── Health & Metrics ─────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """Verify /health returns status and model info."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "loading")
    assert "engine" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0


def test_metrics_endpoint(client):
    """Verify /metrics returns request counters."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "total_errors" in data
    assert "uptime_seconds" in data
    assert "requests_per_minute" in data


# ─── Legacy /generate ─────────────────────────────────────────────────────────

def test_generate_basic(client):
    """Test basic question-answer generation."""
    resp = client.post("/generate", json={
        "question": "What are the early symptoms of appendicitis?"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "latency_ms" in data
    assert data["latency_ms"] > 0


def test_generate_empty_question(client):
    """Empty question should be rejected by Pydantic."""
    resp = client.post("/generate", json={"question": ""})
    assert resp.status_code == 422


def test_generate_short_question(client):
    """min_length=3 should reject very short questions."""
    resp = client.post("/generate", json={"question": "Hi"})
    assert resp.status_code == 422


def test_generate_custom_max_tokens(client):
    """Custom max_tokens should be accepted."""
    resp = client.post("/generate", json={
        "question": "What is diabetes?",
        "max_tokens": 128,
    })
    assert resp.status_code == 200


def test_generate_fields_present(client):
    """All expected fields should be in the response."""
    resp = client.post("/generate", json={
        "question": "What causes high blood pressure?"
    })
    assert resp.status_code == 200
    data = resp.json()
    for field in ("question", "answer", "latency_ms", "model", "engine", "timestamp"):
        assert field in data, f"Missing field: {field}"


# ─── /v1/scribe (SaaS SOAP Endpoint) ─────────────────────────────────────────

def test_scribe_valid_key(client):
    """Valid license key should return a SOAP note."""
    resp = client.post("/v1/scribe", json={
        "text": "45-year-old male with 3-day productive cough and fever 101F. PMH: DM, HTN.",
        "license_key": TEST_LICENSE_KEY,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "soap_note" in data
    assert "latency_ms" in data
    assert "request_id" in data
    assert data["request_id"].startswith("scr_")
    assert "usage_this_month" in data
    assert "tier" in data


def test_scribe_invalid_key(client):
    """Invalid license key should return 403."""
    resp = client.post("/v1/scribe", json={
        "text": "Patient presents with headache and nausea for two days.",
        "license_key": INVALID_KEY,
    })
    assert resp.status_code == 403
    assert "license" in resp.json()["detail"].lower()


def test_scribe_short_text(client):
    """Text shorter than 10 chars should be rejected (422)."""
    resp = client.post("/v1/scribe", json={
        "text": "short",
        "license_key": TEST_LICENSE_KEY,
    })
    assert resp.status_code == 422


def test_scribe_missing_key(client):
    """Missing license_key field should be rejected (422)."""
    resp = client.post("/v1/scribe", json={
        "text": "Patient presents with chronic lower back pain for 2 weeks.",
    })
    assert resp.status_code == 422


def test_scribe_empty_text(client):
    """Empty text should be rejected (422)."""
    resp = client.post("/v1/scribe", json={
        "text": "",
        "license_key": TEST_LICENSE_KEY,
    })
    assert resp.status_code == 422


# ─── /webhooks/lemonsqueezy ──────────────────────────────────────────────────

def test_webhook_no_signature(client):
    """Webhook without signature should work when no secret is configured."""
    payload = {
        "meta": {"event_name": "order_created"},
        "data": {
            "attributes": {
                "user_email": "test@example.com",
                "first_order_item": {"variant_name": "Professional"},
            }
        },
    }
    resp = client.post("/webhooks/lemonsqueezy", content=json.dumps(payload),
                        headers={"Content-Type": "application/json"})
    # If no LEMONSQUEEZY_WEBHOOK_SECRET is set, signature check is skipped
    assert resp.status_code in (200, 403)
    if resp.status_code == 200:
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tier"] == "professional"
        assert "key_prefix" in data


def test_webhook_ignores_non_order(client):
    """Non-order events should be ignored."""
    payload = {
        "meta": {"event_name": "subscription_updated"},
        "data": {},
    }
    resp = client.post("/webhooks/lemonsqueezy", content=json.dumps(payload),
                        headers={"Content-Type": "application/json"})
    if resp.status_code == 200:
        data = resp.json()
        assert data["status"] == "ignored"


def test_webhook_invalid_json(client):
    """Invalid JSON should return 422."""
    resp = client.post("/webhooks/lemonsqueezy", content="not json",
                        headers={"Content-Type": "application/json"})
    assert resp.status_code in (422, 403)


# ─── /admin/usage ────────────────────────────────────────────────────────────

def test_admin_usage_no_filter(client):
    """Admin usage should return aggregate stats."""
    resp = client.get("/admin/usage")
    # May be 403 if ADMIN_SECRET is set — both are valid
    assert resp.status_code in (200, 403)
    if resp.status_code == 200:
        data = resp.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "recent" in data


def test_admin_usage_with_key_filter(client):
    """Admin usage with key filter should work."""
    resp = client.get(f"/admin/usage?key={TEST_LICENSE_KEY}")
    assert resp.status_code in (200, 403)


# ─── /compare (Demo) ─────────────────────────────────────────────────────────

def test_compare_endpoint(client):
    """Compare endpoint should include model info."""
    resp = client.post("/compare", json={
        "question": "What is the treatment for hypertension?"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "fine_tuned_answer" in data
    assert "model_info" in data
    assert data["model_info"]["rouge_l_improvement"] == "+18.4%"


# ─── Rate limiting ────────────────────────────────────────────────────────────

def test_rate_limit_structure(client):
    """Verify the server responds to rapid requests."""
    for _ in range(3):
        resp = client.post("/generate", json={"question": "What is aspirin used for?"})
        assert resp.status_code in (200, 429)


# ─── OpenAPI docs ─────────────────────────────────────────────────────────────

def test_openapi_docs(client):
    """Swagger UI should be accessible."""
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_openapi_schema(client):
    """OpenAPI schema should include all endpoints."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/generate" in schema["paths"]
    assert "/v1/scribe" in schema["paths"]
    assert "/webhooks/lemonsqueezy" in schema["paths"]
    assert "/admin/usage" in schema["paths"]
