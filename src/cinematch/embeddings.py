"""Text embedding helpers.

Full path (default): sentence-transformers + torch, loaded lazily and kept in a
process-global cache so workers share it.

Light path (``CINEMATCH_LIGHT=1``, used by 1 GB Streamlit Cloud deploys):
ONNX via ``fastembed`` — no torch, works under 1 GB RAM.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from cinematch.config import SETTINGS


def _using_light() -> bool:
    return os.getenv("CINEMATCH_LIGHT") == "1"


@lru_cache(maxsize=1)
def _get_model():
    if _using_light():
        from cinematch.lite import get_embedder

        return get_embedder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        SETTINGS.embedding.model_name,
        device=SETTINGS.embedding.device,
    )


def embed_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """Embed a batch of texts -> float32 matrix (n_texts x dim)."""
    if not texts:
        return np.zeros((0, SETTINGS.embedding.dimension), dtype=np.float32)
    model = _get_model()
    batch = batch_size or SETTINGS.embedding.batch_size
    if _using_light():
        vectors = list(model.embed(texts, batch_size=batch))
    else:
        vectors = model.encode(texts, batch_size=batch, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query -> float32 vector of shape (1, dim)."""
    model = _get_model()
    if _using_light():
        vector = next(iter(model.embed([query], batch_size=1)))
    else:
        vector = model.encode(query, show_progress_bar=False)
    return np.asarray([vector], dtype=np.float32)


def model_info() -> dict:
    return {
        "model": SETTINGS.embedding.model_name,
        "dimension": SETTINGS.embedding.dimension,
        "device": SETTINGS.embedding.device,
    }
