FROM python:3.11-slim

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
# running container starts instantly with no network or large downloads:
#   1. MovieLens 100k download + ETL cache
#   2. SVD collaborative model
#   3. Item-based CF + semantic embeddings indexed into embedded Qdrant
RUN python scripts/download_data.py --size 100k \
    && python scripts/train.py \
    && python scripts/index_vectors.py

# Streamlit (proxied by Hugging Face Spaces on app_port) + internal FastAPI
EXPOSE 7860 8000

CMD ["./start.sh"]
