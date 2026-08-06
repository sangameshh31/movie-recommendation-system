"""Content-based scoring: profile vectors over semantic embeddings.

A user profile is the mean embedding of movies they liked (rating >= 4).
Candidates are then retrieved with cosine similarity against that profile
via the vector store.
"""

from __future__ import annotations

import numpy as np

from cinematch.vector_store import VectorStore


class ContentScorer:
    """Retrieves candidates via vector similarity against a user profile."""

    def __init__(self, store: VectorStore):
        self.store = store

    def profile_vector(
        self, liked_ids: list[int], liked_vectors: dict[int, np.ndarray] | None = None
    ) -> np.ndarray | None:
        """Mean embedding of the liked movies; None if nothing liked."""
        if not liked_ids:
            return None
        vectors = liked_vectors
        if vectors is None:
            try:
                vectors = self.store.retrieve_vectors(liked_ids)
            except Exception:
                return None
        present = [v for mid, v in vectors.items() if mid in liked_ids]
        if not present:
            return None
        profile = np.mean(np.stack(present), axis=0).astype(np.float32)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def candidates(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        """Vector-similarity candidates with their cosine scores."""
        return self.store.search(query_vector, top_k=top_k, exclude_ids=exclude_ids)
