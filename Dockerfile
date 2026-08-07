FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    MOVIELENS_SIZE=100k

WORKDIR /app

# Minimal system deps (torch/sentence-transformers ship their own wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt

# Application code
COPY src ./src
COPY scripts ./scripts
COPY frontend ./frontend
COPY tests ./tests
COPY start.sh ./
RUN pip install . && chmod +x start.sh

# Bake the whole offline pipeline into the image at build time, so the
# running container starts instantly with no network or large downloads.
# The enriched processed catalog (with TMDB poster URLs + overviews) is copied
# in, then the SVD model + semantic vector index are rebuilt inside the image.
COPY data/processed/movies.parquet \
     data/processed/ratings.parquet \
     data/processed/tmdb_catalog.parquet \
     ./data/processed/
RUN python scripts/train.py \
    && python scripts/index_vectors.py

# Keep a read-only snapshot of the baked catalog so start.sh can seed a
# persistent volume when DATA_DIR points at one (e.g. HF Spaces /data).
COPY data/processed/movies.parquet \
     data/processed/ratings.parquet \
     data/processed/tmdb_catalog.parquet \
     /opt/cinematch-processed/

# Streamlit (proxied by Hugging Face Spaces on app_port) + internal FastAPI
EXPOSE 7860 8000

CMD ["./start.sh"]
