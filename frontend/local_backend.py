"""In-process backend for Streamlit Cloud (no FastAPI, no Qdrant server).

Mirrors the HTTP API surface the UI already calls (:mod:`cinematch.api`) by
dispatching ``get``/``post`` routes straight to the recommender service inside
the Streamlit process. Runs in "light" mode (``CINEMATCH_LIGHT=1``) so the
1 GB runtime stays under budget: numpy vector search instead of Qdrant, ONNX
embeddings instead of torch, no item-item similarity matrix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Put `src` on the path so `import cinematch` works on Streamlit Cloud
# without installing the package (cwd is the repo root, src is a subdir).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

os.environ.setdefault("CINEMATCH_LIGHT", "1")

import requests  # noqa: E402  (raise errors the UI already catches)


class BackendError(requests.RequestException):
    """Raised for backend failures so existing UI except-blocks still work."""


def _backend_error(message: str) -> BackendError:
    return BackendError(message)


class Backend:
    """Stateless-ish facade over one shared RecommenderService instance."""

    def __init__(self):
        self._service = None

    @property
    def service(self):
        if self._service is None:
            from cinematch.recommend import RecommenderService

            from cinematch.config import SETTINGS

            self._service = RecommenderService(settings=SETTINGS)
        return self._service

    # -- routing ---------------------------------------------------------------

    def get(self, path: str, params: dict | None = None) -> dict:
        params = params or {}
        svc = self.service

        if path == "/health":
            indian = series = anime = 0
            if "origin" in svc.movies.columns:
                indian = int((svc.movies["origin"] == "indian").sum())
                series = int((svc.movies["origin"] == "series").sum())
                anime = int((svc.movies["origin"] == "anime").sum())
            return {
                "status": "ok",
                "vector_index": bool(svc.vector_index_ready),
                "svd_ready": bool(svc._svd is not None),
                "movies": len(svc.movies),
                "indian_movies": indian,
                "series": series,
                "anime": anime,
            }

        if path == "/api/movies/trending":
            return {
                "origin": params.get("origin"),
                "language": params.get("language"),
                "media_type": params.get("media_type"),
                "sort_by": params.get("sort_by", "popularity"),
                "genre": params.get("genre"),
                "results": svc.trending(
                    n=int(params.get("n", 12)),
                    origin=params.get("origin"),
                    language=params.get("language"),
                    media_type=params.get("media_type"),
                    sort_by=params.get("sort_by", "popularity"),
                    genre=params.get("genre"),
                ),
            }

        if path.startswith("/api/movies/search"):
            return {
                "query": params.get("q", ""),
                "origin": params.get("origin"),
                "language": params.get("language"),
                "results": svc.semantic_search(
                    str(params.get("q", "")),
                    n=int(params.get("n", 10)),
                    origin=params.get("origin"),
                    language=params.get("language"),
                ),
            }

        if path.startswith("/api/recommend/") and not path.startswith("/api/recommend/query"):
            user_id = int(path.rsplit("/", 1)[1])
            return {
                "user_id": user_id,
                "recommendations": svc.recommend(
                    user_id,
                    n=int(params.get("n", 10)),
                    with_explanations=params.get("with_explanations", True),
                ),
            }

        if path == "/api/surprise":
            pick = svc.surprise(int(params.get("user_id", 1)))
            if pick is None:
                raise _backend_error("No picks available")
            return {"user_id": params.get("user_id", 1), "pick": pick}

        if path == "/api/people/search":
            person = svc.people.search(str(params.get("q", "")), svc.get_movie)
            if person is None:
                raise _backend_error("Person not found")
            person["catalog_movies"] = person["catalog_movies"][: int(params.get("n", 24))]
            return person

        if path.startswith("/api/user/") and path.endswith("/profile"):
            uid = int(path.split("/")[3])
            ratings = svc.user_ratings(uid, min_rating=float(params.get("min_rating", 4.0)))
            return {
                "user_id": uid,
                "has_history": bool(ratings),
                "liked": ratings,
                "watchlist": sorted(svc.feedback.watchlist(uid)),
                "liked_feedback": sorted(svc.feedback.liked(uid)),
            }

        if path.startswith("/api/user/") and path.endswith("/library"):
            uid = int(path.split("/")[3])
            profile = svc.feedback.profile(uid)
            return {
                "user_id": uid,
                "liked": svc.get_movies(profile["liked"]),
                "disliked": svc.get_movies(profile["disliked"]),
                "watchlist": svc.get_movies(profile["watchlist"]),
                "watched": svc.get_movies(profile["watched"]),
                "stars": {
                    str(mid): value
                    for mid, value in sorted(
                        profile["stars"].items(), key=lambda kv: kv[1], reverse=True
                    )
                },
            }

        if path.startswith("/api/movies/") and path.endswith("/details"):
            movie_id = int(path.split("/")[3])
            item = svc.get_movie(movie_id)
            if item is None:
                raise _backend_error("Movie not found")
            try:
                details = svc.details.enrich(movie_id)
            except Exception as exc:  # pragma: no cover
                from cinematch.details import _empty_payload

                print(f"Details enrichment failed for {movie_id}: {exc}")
                details = _empty_payload("error")
            similar_movies = svc.similar_movies(movie_id, n=int(params.get("similar", 10)))
            return {"movie": item, "details": details, "similar": similar_movies}

        if path.startswith("/api/movies/"):
            movie_id = int(path.rsplit("/", 1)[1])
            item = svc.get_movie(movie_id)
            if item is None:
                raise _backend_error("Movie not found")
            return item

        raise _backend_error(f"Unknown backend route GET {path}")

    def post(self, path: str, body: dict | None = None) -> dict:
        body = body or {}
        svc = self.service

        if path == "/api/feedback":
            svc.feedback.record(
                int(body["user_id"]),
                int(body["movie_id"]),
                str(body["action"]),
                value=body.get("value"),
            )
            return {"status": "recorded", "user_id": body["user_id"], "action": body["action"]}

        if path == "/api/recommend/query":
            return {
                "user_id": body["user_id"],
                "query": body.get("query", ""),
                "recommendations": svc.recommend(
                    int(body["user_id"]),
                    n=int(body.get("n", 10)),
                    query=body.get("query"),
                    with_explanations=body.get("with_explanations", True),
                ),
            }

        if path == "/api/search/refine":
            from cinematch.embeddings import embed_query

            vec = embed_query(str(body.get("seed", "")).strip())
            nudge = None
            for phrase in body.get("additions", []):
                if phrase and phrase.strip():
                    nudge = (
                        embed_query(phrase.strip())
                        if nudge is None
                        else nudge + embed_query(phrase.strip())
                    )
            for phrase in body.get("removals", []):
                if phrase and phrase.strip():
                    nudge = (
                        -embed_query(phrase.strip())
                        if nudge is None
                        else nudge - embed_query(phrase.strip())
                    )
            if nudge is not None:
                vec = vec + 0.6 * nudge
            return {
                "seed": body.get("seed"),
                "additions": body.get("additions", []),
                "removals": body.get("removals", []),
                "results": svc.semantic_search_vector(
                    vec,
                    n=int(body.get("n", 12)),
                    origin=body.get("origin"),
                    language=body.get("language"),
                ),
            }

        raise _backend_error(f"Unknown backend route POST {path}")


def get_backend() -> Backend:
    return Backend()
