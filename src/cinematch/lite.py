"""Lightweight runtime for 1 GB Streamlit Cloud deploys.

Replaces the Qdrant vector store + sentence-transformers (torch) stack with
plain numpy + ONNX (via ``fastembed``), loading the precomputed semantic index
from ``data/processed/light_index.pkl``. Keeps the exact duck-typed interface
of :class:`cinematch.vector_store.VectorStore` so the recommender service is
unchanged.
"""

from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np

from cinematch.config import SETTINGS


@lru_cache(maxsize=1)
def load_light_index() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict[int, int]]:
    """Load (ids, vectors, normalized, payloads, id->row) once per process."""
    path = SETTINGS.paths.processed_dir / "light_index.pkl"
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    ids = np.asarray(data["ids"], dtype=np.int64)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    normalized = vectors / norms[:, None]
    id_to_row = {int(mid): i for i, mid in enumerate(ids)}
    return ids, vectors, normalized, data["payloads"], id_to_row


class LightVectorStore:
    """Duck-typed replacement for :class:`cinematch.vector_store.VectorStore`."""

    def __init__(self, config=None):
        self.config = config or SETTINGS.qdrant
        self._ids, self._vectors, self._normalized, self._payloads, self._row = (
            load_light_index()
        )

    # -- collection management -------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        return None

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError("light index is immutable at runtime")

    def count(self) -> int:
        return len(self._ids)

    # -- retrieval ---------------------------------------------------------------

    def retrieve_vectors(self, ids: list[int]) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for mid in ids:
            row = self._row.get(int(mid))
            if row is not None:
                out[int(mid)] = self._vectors[row]
        return out

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        exclude_ids: set[int] | None = None,
        origin: str | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """Cosine-similarity search. Returns payload dicts with a ``score``."""
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        qn = q / norm if norm > 0 else q
        scores = self._normalized @ qn
        order = np.argsort(-scores)

        out: list[dict] = []
        for idx in order:
            payload = self._payloads[idx]
            mid = int(payload["movie_id"])
            if exclude_ids and mid in exclude_ids:
                continue
            if origin and payload.get("origin") != origin:
                continue
            if language and payload.get("language") != language:
                continue
            item = dict(payload)
            item["score"] = float(scores[idx])
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def close(self) -> None:
        return None


def get_embedder():
    """ONNX-backed text embedder (fastembed) — no torch, ~80 MB model."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=SETTINGS.embedding.model_name)
