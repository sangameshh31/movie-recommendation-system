"""Recommendation orchestrator.

Wires ingestion -> models -> vector store -> feedback -> explanations into a
single service used by the API and Streamlit UI. Implements the two-stage
pipeline: candidate generation (CF + content paths) then hybrid re-ranking.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from cinematch.config import SETTINGS
from cinematch.data import load_processed
from cinematch.embeddings import embed_query
from cinematch.explainer import explain
from cinematch.feedback import FeedbackStore
from cinematch.models import SVDFactorizer, ItemBasedCF, hybrid_rerank, minmax_normalize, popularity_scores
from cinematch.models.content import ContentScorer
from cinematch.vector_store import VectorStore


class RecommenderService:
    def __init__(self, settings=SETTINGS, lazy: bool = False):
        self.settings = settings
        self.feedback = FeedbackStore()
        self._svd: SVDFactorizer | None = None
        self._itemcf: ItemBasedCF | None = None
        self._store: VectorStore | None = None
        self._content: ContentScorer | None = None
        self._popularity: dict[int, float] = {}
        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self._svd_path = (
            settings.paths.models_dir / f"svd_{settings.data.size}.pkl"
        )
        if not lazy:
            self.load()

    # -- lifecycle ---------------------------------------------------------------

    def load(self) -> "RecommenderService":
        data = load_processed(paths=self.settings.paths)
        self.movies, self.ratings = data["movies"], data["ratings"]
        self._load_or_fit_svd()
        self._itemcf = ItemBasedCF().fit(self.ratings)
        self._popularity = popularity_scores(self.ratings)
        self._vote_avg: dict[int, float] = {}
        self._year_of: dict[int, int] = {}
        if "vote_average" in self.movies.columns:
            self._vote_avg = {
                int(r.movie_id): float(r.vote_average)
                for r in self.movies.itertuples(index=False)
                if pd.notna(r.vote_average)
            }
        self._year_of = {
            int(r.movie_id): (int(r.year) if pd.notna(r.year) else 0)
            for r in self.movies.itertuples(index=False)
        }
        return self

    def _load_or_fit_svd(self) -> None:
        if self._svd_path.exists():
            self._svd = SVDFactorizer.load(self._svd_path)
            return
        self._svd = SVDFactorizer(
            n_components=self.settings.retrieval.svd_factors,
            seed=self.settings.data.random_seed,
        ).fit(self.ratings)
        self._svd.save(self._svd_path)
        print(
            f"Trained SVD: {self._svd.U.shape[0]:,} users x "
            f"{self._svd.Vt.shape[1]:,} movies "
            f"(explained variance {self._svd.explained_variance:.2%})"
        )

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(config=self.settings.qdrant)
            self._content = ContentScorer(self._store)
        return self._store

    @property
    def content(self) -> ContentScorer:
        if self._content is None:
            self.store  # ensure store + content built
        return self._content

    @property
    def vector_index_ready(self) -> bool:
        try:
            return self.store.count() > 0
        except Exception:
            return False

    # -- lookups ------------------------------------------------------------------

    def get_movie(self, movie_id: int) -> dict | None:
        row = self.movies[self.movies["movie_id"] == movie_id]
        if row.empty:
            return None
        return self._movie_payload(row.iloc[0])

    def get_movies(self, movie_ids: Iterable[int]) -> list[dict]:
        ids = list(dict.fromkeys(movie_ids))
        df = self.movies[self.movies["movie_id"].isin(ids)]
        order = {mid: i for i, mid in enumerate(ids)}
        df = df.sort_values("movie_id", key=lambda s: s.map(order))
        return [self._movie_payload(r) for r in df.itertuples(index=False)]

    def _movie_payload(self, row) -> dict:
        poster = getattr(row, "poster_url", None)
        poster_url = None
        if (
            poster is not None
            and not (isinstance(poster, float) and pd.isna(poster))
            and str(poster).strip()
        ):
            poster_url = str(poster)
        vote = getattr(row, "vote_average", None)
        vote_avg = None
        if vote is not None and pd.notna(vote):
            vote_avg = round(float(vote), 1)
        return {
            "movie_id": int(row.movie_id),
            "title": row.title,
            "clean_title": row.clean_title,
            "year": None if pd.isna(row.year) else int(row.year),
            "genres": list(row.genres),
            "language": str(getattr(row, "language", "") or ""),
            "origin": str(getattr(row, "origin", "") or ""),
            "media_type": str(getattr(row, "media_type", "") or "movie"),
            "vote_average": vote_avg,
            "poster_url": poster_url,
        }

    # -- user profile --------------------------------------------------------------

    def user_has_history(self, user_id: int) -> bool:
        return user_id in set(self.ratings["user_id"].unique())

    def user_ratings(self, user_id: int, min_rating: float = 0) -> list[dict]:
        sub = self.ratings[
            (self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= min_rating)
        ]
        if sub.empty:
            return []
        merged = sub.merge(self.movies[["movie_id", "title", "genres", "year"]], on="movie_id")
        merged = merged.sort_values("rating", ascending=False)
        return [
            {
                "movie_id": int(r.movie_id),
                "title": r.title,
                "rating": float(r.rating),
                "genres": list(r.genres),
                "year": None if pd.isna(r.year) else int(r.year),
            }
            for r in merged.itertuples(index=False)
        ]

    def liked_movies(self, user_id: int, limit: int = 5) -> list[dict]:
        return self.user_ratings(user_id, min_rating=4.0)[:limit]

    # -- retrieval -----------------------------------------------------------------

    def _cf_candidates(
        self, user_id: int, exclude: set[int]
    ) -> dict[int, float]:
        """Blend SVD (latent factors) and item-based CF (co-rating similarity).

        Both paths are min-max normalized on their own scale before the
        weighted union, so the blend stays comparable inside the hybrid ranker.
        """
        if self._svd is None or self._itemcf is None or not self.user_has_history(user_id):
            return {}
        n = self.settings.retrieval.n_candidates
        svd = dict(
            self._svd.top_for_user(
                user_id, k=n, exclude=exclude
            )
        )
        icf = dict(
            self._itemcf.top_for_user(
                user_id, k=n, exclude=exclude
            )
        )
        if not svd and not icf:
            return {}

        w_svd, w_icf = 0.4, 0.6
        norm_svd = minmax_normalize(svd)
        norm_icf = minmax_normalize(icf)
        return {
            mid: w_svd * norm_svd.get(mid, 0.0) + w_icf * norm_icf.get(mid, 0.0)
            for mid in set(svd) | set(icf)
        }

    def _content_candidates(
        self, query_vector: np.ndarray, exclude: set[int]
    ) -> dict[int, float]:
        if not self.vector_index_ready:
            return {}
        hits = self.content.candidates(
            query_vector,
            top_k=self.settings.retrieval.n_candidates,
            exclude_ids=exclude or None,
        )
        return {int(h["movie_id"]): float(h["score"]) for h in hits}

    def _build_content_vector(
        self, user_id: int, query: str | None
    ) -> np.ndarray | None:
        if query and query.strip():
            return embed_query(query.strip())
        if not self.vector_index_ready:
            return None
        liked = self.liked_movies(user_id)
        if liked:
            try:
                return self.content.profile_vector([m["movie_id"] for m in liked])
            except Exception:
                return None
        return None

    def recommend(
        self,
        user_id: int,
        n: int | None = None,
        query: str | None = None,
        with_explanations: bool = False,
    ) -> list[dict]:
        """Two-stage hybrid recommendation for a user.

        * Stage 1 (candidates): CF path (SVD) + content path (vector store).
        * Stage 2 (re-rank): weighted hybrid score + feedback deltas.
        """
        cfg = self.settings.retrieval
        n = n or cfg.n_recs
        exclude = {m["movie_id"] for m in self.user_ratings(user_id)}

        cf = self._cf_candidates(user_id, exclude)
        vec = self._build_content_vector(user_id, query)
        cb = self._content_candidates(vec, exclude) if vec is not None else {}

        candidates: dict[int, dict] = {}
        for mid in set(cf) | set(cb):
            candidates[mid] = {
                "cf": cf.get(mid, 0.0),
                "cb": cb.get(mid, 0.0),
                "pop": self._popularity.get(mid, 0.0),
            }

        # Diversity injection: let trending Indian blockbusters compete in the
        # pool so the Indian catalog is visible even without an explicit query.
        for item in self.trending(origin="indian", n=cfg.n_candidates // 2):
            mid = item["movie_id"]
            if mid not in candidates:
                candidates[mid] = {
                    "cf": 0.0,
                    "cb": 0.0,
                    "pop": item["score"],
                }

        if not candidates:
            # Cold start with no query: fall back to pure popularity.
            ranked = sorted(self._popularity.items(), key=lambda kv: kv[1], reverse=True)[:n]
            movie_ids = [mid for mid, _ in ranked]
            return self._finalize(movie_ids, user_id, with_explanations, scores=dict(ranked))

        feedback_deltas = self.feedback.genre_overlap_deltas(user_id, self.movies)
        for mid, delta in feedback_deltas.items():
            if mid in candidates:
                candidates[mid]["feedback"] = delta

        weights = {
            "cf": cfg.weight_cf,
            "cb": cfg.weight_cb,
            "pop": cfg.weight_pop,
            "feedback": cfg.feedback_boost,
        }
        ranked = hybrid_rerank(candidates, weights)
        movie_ids = [mid for mid, _ in ranked[:n]]
        scores = dict(ranked)
        return self._finalize(movie_ids, user_id, with_explanations, scores=scores)

    def semantic_search(
        self,
        query: str,
        n: int = 10,
        origin: str | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """Natural-language vector search over the movie catalog."""
        if not self.vector_index_ready:
            raise RuntimeError(
                "Vector index is not built. Run `python scripts/index_vectors.py` first."
            )
        vec = embed_query(query.strip())
        hits = self.store.search(
            vec, top_k=n, origin=origin or None, language=language or None
        )
        return [
            {
                "movie_id": int(h["movie_id"]),
                "title": h["title"],
                "clean_title": h["clean_title"],
                "year": h.get("year"),
                "genres": h.get("genres", []),
                "language": h.get("language", ""),
                "origin": h.get("origin", ""),
                "media_type": h.get("media_type", "movie"),
                "vote_average": h.get("vote_average"),
                "poster_url": h.get("poster_url") or None,
                "similarity": round(float(h["score"]), 4),
            }
            for h in hits
        ]

    def trending(
        self,
        n: int = 10,
        origin: str | None = None,
        language: str | None = None,
        media_type: str | None = None,
        sort_by: str = "popularity",
        genre: str | None = None,
    ) -> list[dict]:
        """Top titles by popularity / rating / recency, optionally filtered.

        ``origin`` selects a catalog partition (``indian``, ``anime``,
        ``series``, ``new``, ``movielens``), ``media_type`` one of
        ``movie``/``series``, ``sort_by`` one of ``popularity`` / ``rating`` /
        ``new``, and ``genre`` a single genre to require.
        """
        df = self.movies
        if origin:
            df = df[df["origin"] == origin]
        if media_type:
            df = df[df["media_type"] == media_type]
        if language:
            df = df[df["language"] == language]
        if genre:
            df = df[
                df["genres"].apply(
                    lambda g: g is not None and isinstance(g, (list, tuple, np.ndarray)) and genre in g
                )
            ]

        if sort_by == "rating":
            key = lambda mid: (self._vote_avg.get(mid, 0.0), self._popularity.get(mid, 0.0))
        elif sort_by == "new":
            key = lambda mid: (self._year_of.get(mid, 0), self._popularity.get(mid, 0.0))
        else:
            key = lambda mid: self._popularity.get(mid, 0.0)

        ranked = sorted(df["movie_id"].tolist(), key=key, reverse=True)[:n]
        out = []
        for mid in ranked:
            item = self.get_movie(mid)
            if item is not None:
                item["score"] = round(float(self._popularity.get(mid, 0.0)), 4)
                out.append(item)
        return out

    # -- output ----------------------------------------------------------------------

    def _finalize(
        self,
        movie_ids: list[int],
        user_id: int,
        with_explanations: bool,
        scores: dict[int, float],
    ) -> list[dict]:
        movies = self.get_movies(movie_ids)
        liked = self.liked_movies(user_id) if with_explanations else []
        for m in movies:
            m["score"] = round(scores.get(m["movie_id"], 0.0), 4)
            if with_explanations:
                m["why"] = explain(m, liked)
        return movies
