#!/usr/bin/env bash
# CineMatch AI container entrypoint.
# Runs the FastAPI backend on :8000 (internal) and the Streamlit UI on :7860
# (the port Hugging Face Spaces proxies to the internet).
set -e
cd /app

# If DATA_DIR points at a persistent volume (e.g. HF Spaces mounts /data),
# seed it with the baked catalog so the app reads/writes there instead of the
# ephemeral container filesystem.
if [ -n "${DATA_DIR:-}" ] && [ -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR/processed"
    if [ ! -f "$DATA_DIR/processed/movies.parquet" ]; then
        echo "[CineMatch] seeding baked catalog into $DATA_DIR/processed ..."
        cp /opt/cinematch-processed/*.parquet "$DATA_DIR/processed/"
    fi
fi

echo "[CineMatch] starting API on :8000 ..."
python scripts/run_api.py --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "[CineMatch] starting Streamlit UI on :7860 ..."
streamlit run frontend/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true

kill $API_PID 2>/dev/null || true
