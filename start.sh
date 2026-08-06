#!/usr/bin/env bash
# CineMatch AI container entrypoint.
# Runs the FastAPI backend on :8000 (internal) and the Streamlit UI on :7860
# (the port Hugging Face Spaces proxies to the internet).
set -e
cd /app

echo "[CineMatch] starting API on :8000 ..."
python scripts/run_api.py --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "[CineMatch] starting Streamlit UI on :7860 ..."
streamlit run frontend/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true

kill $API_PID 2>/dev/null || true
