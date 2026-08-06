"""Qdrant vector-store wrapper.

Works in two modes:

* **Embedded (default):** persistent local storage under ``qdrant_storage/``,
  no server or Docker required.
* **Remote:** connect to a Docker/cloud Qdrant when ``QDRANT_URL`` is set.

Payloads carry full movie metadata so the store can answer retrieval requests
without a second round-trip to the database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from cinematch.config import SETTINGS


class VectorStore:
    def __init__(self, config=SETTINGS.qdrant):
        self.config = config
        if config.local_mode:
            SETTINGS.paths.qdrant_dir.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(SETTINGS.paths.qdrant_dir))
        else:
            self._client = QdrantClient(url=config.url, api_key=config.api_key or None)

    # -- collection management ------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        existing = self._client.get_collections().collections
        names = {c.name for c in existing}
        if self.config.collection not in names:
            self._client.create_collection(
                collection_name=self.config.collection,
                vectors_config=qm.VectorParams(
                    size=dimension, distance=qm.Distance.COSINE
                ),
            )

    def reset(self) -> None:
        self._client.recreate_collection(
            collection_name=self.config.collection,
            vectors_config=qm.VectorParams(
                size=SETTINGS.embedding.dimension, distance=qm.Distance.COSINE
            ),
        )

    def count(self) -> int:
        return self._client.count(self.config.collection).count

    # -- indexing --------------------------------------------------------------

    def index_movies(self, movies: pd.DataFrame, vectors: np.ndarray) -> int:
        """Upsert movie vectors + metadata payload. Returns number of points."""
        self.ensure_collection(vectors.shape[1])
        ids, payloads = [], []
        for row in movies.itertuples(index=False):
            ids.append(int(row.movie_id))
            payloads.append(
                {
                    "movie_id": int(row.movie_id),
                    "title": row.title,
                    "clean_title": row.clean_title,
                    "year": None if pd.isna(row.year) else int(row.year),
                    "genres": list(row.genres),
                }
            )

        self._client.upsert(
            collection_name=self.config.collection,
            points=qm.Batch(ids=ids, vectors=vectors.tolist(), payloads=payloads),
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
    ) -> list[dict]:
        """Cosine-similarity search. Returns payload dicts with a `score`."""
        flat = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        qfilter = None
        if exclude_ids:
            qfilter = qm.Filter(
                must_not=[
                    qm.FieldCondition(
                        key="movie_id", match=qm.MatchAny(any=list(exclude_ids))
                    )
                ]
            )
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
