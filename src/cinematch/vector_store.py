"""Qdrant vector-store wrapper.

Works in two modes:

* **Embedded (default):** persistent local storage under ``qdrant_storage/``,
  no server or Docker required.
* **Remote:** connect to a Docker/cloud Qdrant when ``QDRANT_URL`` is set.

Payloads carry full movie metadata so the store can answer retrieval requests
without a second round-trip to the database.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from cinematch.config import SETTINGS


class VectorStore:
    def __init__(self, config=SETTINGS.qdrant):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm  # noqa: F401 (import guard)

        self._qm = qm
        self.config = config
        if config.local_mode:
            SETTINGS.paths.qdrant_dir.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(SETTINGS.paths.qdrant_dir))
        else:
            self._client = QdrantClient(url=config.url, api_key=config.api_key or None)
        # The collection is static at runtime, so cache the point count to avoid
        # re-reading index metadata from disk on every health check / candidate
        # generation (disk reads are the source of multi-second cold stalls).
        self._count_cache: int | None = None
        self._count_cached_at: float = 0.0

    # -- collection management ------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        existing = self._client.get_collections().collections
        names = {c.name for c in existing}
        if self.config.collection not in names:
            self._client.create_collection(
                collection_name=self.config.collection,
                vectors_config=self._qm.VectorParams(
                    size=dimension, distance=self._qm.Distance.COSINE
                ),
            )

    def reset(self) -> None:
        self._client.recreate_collection(
            collection_name=self.config.collection,
            vectors_config=self._qm.VectorParams(
                size=SETTINGS.embedding.dimension, distance=self._qm.Distance.COSINE
            ),
        )
        self._count_cache = None

    def count(self) -> int:
        now = time.time()
        if self._count_cache is not None and now - self._count_cached_at < 60.0:
            return self._count_cache
        value = self._client.count(self.config.collection).count
        self._count_cache = value
        self._count_cached_at = now
        return value

    # -- indexing --------------------------------------------------------------

    def index_movies(self, movies: pd.DataFrame, vectors: np.ndarray) -> int:
        """Upsert movie vectors + metadata payload. Returns number of points."""
        self.ensure_collection(vectors.shape[1])
        ids, payloads = [], []
        for row in movies.itertuples(index=False):
            poster = getattr(row, "poster_url", "")
            poster_url = (
                ""
                if poster is None
                or (isinstance(poster, float) and pd.isna(poster))
                or not str(poster).strip()
                else str(poster)
            )
            ids.append(int(row.movie_id))
            vote = getattr(row, "vote_average", None)
            payloads.append(
                {
                    "movie_id": int(row.movie_id),
                    "title": row.title,
                    "clean_title": row.clean_title,
                    "year": None if pd.isna(row.year) else int(row.year),
                    "genres": list(row.genres),
                    "language": str(getattr(row, "language", "") or ""),
                    "origin": str(getattr(row, "origin", "") or ""),
                    "media_type": str(getattr(row, "media_type", "") or "movie"),
                    "vote_average": (
                        None
                        if vote is None or pd.isna(vote)
                        else round(float(vote), 1)
                    ),
                    "poster_url": poster_url,
                }
            )

        self._client.upsert(
            collection_name=self.config.collection,
            points=self._qm.Batch(ids=ids, vectors=vectors.tolist(), payloads=payloads),
        )
        return len(ids)

    # -- retrieval ----------------------------------------------------------------

    def retrieve_vectors(self, ids: list[int]) -> dict[int, np.ndarray]:
        """Fetch raw vectors for the given movie ids (used to build profiles)."""
        points = self._client.retrieve(
            collection_name=self.config.collection, ids=ids, with_vectors=True
        )
        return {p.id: np.asarray(p.vector, dtype=np.float32) for p in points}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        exclude_ids: set[int] | None = None,
        origin: str | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """Cosine-similarity search. Returns payload dicts with a `score`."""
        flat = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        must_not: list[self._qm.FieldCondition] = []
        if exclude_ids:
            must_not.append(
                self._qm.FieldCondition(
                    key="movie_id", match=self._qm.MatchAny(any=list(exclude_ids))
                )
            )
        must: list[self._qm.FieldCondition] = []
        if origin:
            must.append(
                self._qm.FieldCondition(key="origin", match=self._qm.MatchValue(value=origin))
            )
        if language:
            must.append(
                self._qm.FieldCondition(key="language", match=self._qm.MatchValue(value=language))
            )
        qfilter = None
        if must or must_not:
            qfilter = self._qm.Filter(must=must or None, must_not=must_not or None)
        hits = self._client.query_points(
            collection_name=self.config.collection,
            query=flat.tolist(),
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        ).points
        out = []
        for hit in hits:
            item = dict(hit.payload)
            item["score"] = hit.score
            out.append(item)
        return out

    def close(self) -> None:
        self._client.close()
