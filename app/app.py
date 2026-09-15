"""CallSense AI dashboard entrypoint. Run with: streamlit run app/app.py

Dashboard pages (call analysis, risk view, agent analytics, search) land
here module by module as the corresponding pipeline stage is built — see
docs/PROJECT_PLAN.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from configs.settings import get_settings

st.set_page_config(page_title="CallSense AI", layout="wide")

settings = get_settings()

st.title("CallSense AI")
st.caption("Customer Conversation Analytics & Quality Intelligence Platform")

api_url = f"http://{settings.api_host if settings.api_host != '0.0.0.0' else 'localhost'}:{settings.api_port}"

try:
    resp = requests.get(f"{api_url}/health", timeout=2)
    status = resp.json().get("status") if resp.ok else f"error ({resp.status_code})"
except requests.RequestException:
    status = "unreachable"

col1, col2 = st.columns(2)
col1.metric("Backend status", status)
col2.metric("Environment", settings.app_env)

st.info(
    "Call upload, transcript view, sentiment/emotion timeline, risk scoring, "
    "agent analytics, and semantic search will each land here as their "
    "pipeline module is completed."
)
