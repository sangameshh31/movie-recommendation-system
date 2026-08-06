"""Text embedding helpers built on sentence-transformers.

The model is loaded lazily and kept in a process-global cache so API workers
share it instead of re-downloading per request.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from cinematch.config import SETTINGS


@lru_cache(maxsize=1)
def _get_model():
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
    vectors = model.encode(texts, batch_size=batch, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query -> float32 vector of shape (1, dim)."""
    model = _get_model()
    vector = model.encode(query, show_progress_bar=False)
    return np.asarray([vector], dtype=np.float32)


def model_info() -> dict:
    return {
        "model": SETTINGS.embedding.model_name,
        "dimension": SETTINGS.embedding.dimension,
        "device": SETTINGS.embedding.device,
    }
