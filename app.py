"""
MedScribe AI — Streamlit Frontend
===================================
License-gated clinical dictation → SOAP note converter.
Connects to FastAPI backend at /v1/scribe.

Usage:
    streamlit run app.py
"""

import os
import time
import html
import logging
import streamlit as st
import requests
from dotenv import load_dotenv

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SCRIBE_ENDPOINT = f"{API_BASE_URL}/v1/scribe"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("medscribe-ui")


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedScribe AI — Clinical SOAP Notes",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        max-width: 780px;
        padding-top: 2rem;
    }

    /* Header */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e40af, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .soap-card {
        background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
        white-space: pre-wrap;
        font-size: 0.95rem;
    }
    .soap-card h4 {
        color: #1e40af;
        margin: 0.8rem 0 0.3rem 0;
    }

    .status-bar {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.5rem;
    }

    .license-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stTextArea"] textarea {
        border: 2px solid #cbd5e1;
        border-radius: 10px;
        font-size: 0.95rem;
        min-height: 180px;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
    }

    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(59,130,246,0.4);
    }

    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def check_backend_health() -> bool:
    """Ping the backend /health endpoint to verify connectivity."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def validate_license_key(license_key: str) -> dict:
    """Validate a license key against the backend without consuming a note."""
    validate_url = f"{API_BASE_URL}/v1/scribe"
    payload = {"text": "VALIDATE_KEY_ONLY — this is a license validation ping request", "license_key": license_key}
    try:
        resp = requests.post(validate_url, json=payload, timeout=8)
        if resp.status_code == 403:
            return {"valid": False, "error": "Invalid or expired license key."}
        # Any non-403 means the key is accepted (even 503/500 = key is valid, server issue)
        return {"valid": True}
    except requests.ConnectionError:
        return {"valid": None, "error": "Server unavailable — key stored for later."}
    except requests.RequestException:
        return {"valid": None, "error": "Could not verify — key stored for later."}


def call_scribe_api(clinical_text: str, license_key: str) -> dict:
    """POST clinical text to /v1/scribe and return the response dict."""
    payload = {"text": clinical_text, "license_key": license_key}
    try:
        resp = requests.post(SCRIBE_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 403:
            return {"error": "Invalid or expired license key. Please check your key and try again."}
        if resp.status_code == 422:
            detail = resp.json().get("detail", "Invalid input.")
            return {"error": f"Validation error: {detail}"}
        if resp.status_code == 504:
            return {"error": "Server timeout. The model is taking too long — please try again."}
        if resp.status_code == 429:
            detail = resp.json().get("detail", "Rate limit exceeded.")
            return {"error": detail}
        if resp.status_code != 200:
            return {"error": f"Server error ({resp.status_code}). Please try again later."}
        return resp.json()
    except requests.Timeout:
        return {"error": "Request timed out. The server may be under heavy load."}
    except requests.ConnectionError:
        return {"error": "Cannot reach the MedScribe server. Please check your connection."}
    except requests.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


# ─── Session State ────────────────────────────────────────────────────────────

if "license_validated" not in st.session_state:
    st.session_state.license_validated = False
if "license_key" not in st.session_state:
    st.session_state.license_key = ""
if "soap_result" not in st.session_state:
    st.session_state.soap_result = None
if "history" not in st.session_state:
    st.session_state.history = []


# ─── UI: Header ──────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🩺 MedScribe AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Convert clinical dictations into structured SOAP notes in seconds</div>', unsafe_allow_html=True)


# ─── UI: License Key Gate ────────────────────────────────────────────────────

if not st.session_state.license_validated:
    st.markdown('<div class="license-box">', unsafe_allow_html=True)
    st.markdown("#### 🔑 Enter Your License Key")
    st.caption("Purchase a license at [medscribe.ai](https://medscribe.ai) to get started.")

    key_input = st.text_input(
        "License Key",
        type="password",
        placeholder="MSAI-XXXX-XXXX-XXXX",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔓 Activate License", use_container_width=True):
            if not key_input or len(key_input.strip()) < 8:
                st.error("Please enter a valid license key.")
            else:
                with st.spinner("Validating license..."):
                    result = validate_license_key(key_input.strip())
                    if result.get("valid") is False:
                        st.error(f"❌ {result['error']}")
                    elif result.get("valid") is None:
                        st.warning(f"⚠️ {result['error']}")
                        st.session_state.license_key = key_input.strip()
                        st.session_state.license_validated = True
                        st.rerun()
                    else:
                        st.session_state.license_key = key_input.strip()
                        st.session_state.license_validated = True
                        st.success("✅ License activated!")
                        time.sleep(0.5)
                        st.rerun()
    with col2:
        backend_ok = check_backend_health()
        if backend_ok:
            st.markdown("🟢 Server online")
        else:
            st.markdown("🔴 Server offline")
            
    st.markdown("---")
    st.markdown("#### ✨ Don't have a key?")
    trial_email = st.text_input("Email address", placeholder="doctor@clinic.com", key="trial_email")
    if st.button("🎁 Start 3-Day Free Trial", use_container_width=True):
        if not trial_email or "@" not in trial_email:
            st.error("Please enter a valid email address.")
        else:
            with st.spinner("Generating trial key..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/v1/trial", json={"email": trial_email}, timeout=5)
                    if res.status_code == 200:
                        data = res.json()
                        st.success("✅ " + data["message"])
                        st.info(f"Your trial key: **{data['license_key']}**\n\n(It will expire on {data['expires_at'][:10]})")
                    else:
                        st.error(f"Failed to generate trial key: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to server: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ─── UI: Main App (License Validated) ────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Key:** `{st.session_state.license_key[:8]}...`")

    if check_backend_health():
        st.success("🟢 Backend connected")
    else:
        st.error("🔴 Backend offline")

    st.divider()

    if st.button("🔒 Logout", use_container_width=True):
        st.session_state.license_validated = False
        st.session_state.license_key = ""
        st.session_state.soap_result = None
        st.rerun()

    st.divider()
    st.caption("MedScribe AI v1.0")
    st.caption("Powered by Mistral-7B (QLoRA)")

# Main input
st.markdown("### 📝 Clinical Dictation")
st.caption("Paste or type the patient encounter notes below.")

clinical_text = st.text_area(
    "Clinical text",
    height=200,
    placeholder=(
        "Example: 45-year-old male presents with 3-day history of productive cough, "
        "fever 101.2F, and right-sided chest pain worse with deep inspiration. "
        "PMH: Type 2 DM, HTN. Meds: Metformin 1000mg BID, Lisinopril 20mg daily. "
        "Vitals: BP 138/88, HR 96, RR 22, SpO2 94% on RA. Lung exam reveals "
        "decreased breath sounds and crackles in right lower lobe..."
    ),
    label_visibility="collapsed",
)

# Submit
if st.button("🔬 Generate SOAP Note", use_container_width=True):
    if not clinical_text or len(clinical_text.strip()) < 10:
        st.warning("⚠️ Please enter at least 10 characters of clinical text.")
    else:
        with st.spinner("🧠 Analyzing clinical dictation..."):
            result = call_scribe_api(clinical_text.strip(), st.session_state.license_key)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.session_state.soap_result = result
            st.session_state.history.append({
                "input": clinical_text[:80] + "...",
                "timestamp": time.strftime("%H:%M:%S"),
                "latency": result.get("latency_ms", 0),
            })

# ─── Output Panel ─────────────────────────────────────────────────────────────

if st.session_state.soap_result:
    result = st.session_state.soap_result
    soap_note = result.get("soap_note", result.get("answer", "No output returned."))

    st.markdown("### 📋 SOAP Note")
    # Escape HTML to prevent XSS, then render safely
    safe_note = html.escape(soap_note).replace("\n", "<br>")
    st.markdown(f'<div class="soap-card">{safe_note}</div>', unsafe_allow_html=True)

    # Metadata bar
    latency = result.get("latency_ms", 0)
    req_id = html.escape(str(result.get("request_id", "N/A")))
    tier = html.escape(str(result.get("tier", "")))
    usage = result.get("usage_this_month", 0)
    st.markdown(
        f'<div class="status-bar">'
        f'⚡ {latency:.0f}ms &nbsp;|&nbsp; 🆔 {req_id}'
        f' &nbsp;|&nbsp; 📊 {usage} notes this month'
        f' &nbsp;|&nbsp; 🏷️ {tier.title()}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Copy to clipboard + Clear
    col_copy, col_clear = st.columns(2)
    with col_copy:
        st.code(soap_note, language=None)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.soap_result = None
            st.rerun()

# ─── History ──────────────────────────────────────────────────────────────────

if st.session_state.history:
    with st.expander(f"📜 Session History ({len(st.session_state.history)} notes)"):
        for i, h in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**{i}.** {h['input']} — `{h['timestamp']}` ({h['latency']:.0f}ms)")
