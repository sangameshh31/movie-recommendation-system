"""FastAPI backend for CineMatch AI.

Run with:  uvicorn cinematch.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cinematch.config import SETTINGS
from cinematch.embeddings import model_info
from cinematch.recommend import RecommenderService


class RecommendQueryRequest(BaseModel):
    user_id: int = Field(..., description="User id from the MovieLens dataset")
    query: str = Field(..., min_length=1, description="Natural-language query")
    n: int = Field(10, ge=1, le=50)
    with_explanations: bool = True


class FeedbackRequest(BaseModel):
    user_id: int
    movie_id: int
    action: str = Field(
        ..., description="One of: like | dislike | watchlist | remove"
    )


service: RecommenderService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("Loading CineMatch models ...")
    service = RecommenderService(settings=SETTINGS)
    # Warm the lazy one-time paths (embedded Qdrant init + embedding model) so
    # the first user request doesn't pay the 2-4s cold-start penalty.
    try:
        from cinematch.embeddings import embed_query

        vector = embed_query("warmup")
        service.content.candidates(vector, top_k=10)
        service.content.profile_vector([int(service.movies["movie_id"].iloc[0])])
    except Exception as exc:  # pragma: no cover
        print(f"Warmup (embedding model) skipped: {exc}")
    try:
        # Probe the optional Ollama endpoint once at startup; its first call
        # can hang for seconds on Windows if the port is firewalled/dropped.
        from cinematch.explainer import _ollama_available

        _ollama_available()
    except Exception as exc:  # pragma: no cover
        print(f"Warmup (ollama probe) skipped: {exc}")
    try:
        _ = service.vector_index_ready
    except Exception as exc:  # pragma: no cover
        print(f"Warmup (vector index) skipped: {exc}")
    print(
        f"Ready: {len(service.movies):,} movies | "
        f"vectors indexed={service.vector_index_ready}"
    )
    yield
    service = None


app = FastAPI(
    title="CineMatch AI",
    version="0.1.0",
    description="Hybrid movie recommendation engine (CF + content + LLM).",
    lifespan=lifespan,
)


def _require_service() -> RecommenderService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is still loading")
    return service


@app.get("/health")
def health():
    svc = service
    indian = 0
    if svc is not None and "origin" in svc.movies.columns:
        indian = int((svc.movies["origin"] == "indian").sum())
    return {
        "status": "ok" if svc is not None else "loading",
        "vector_index": bool(svc and svc.vector_index_ready),
        "svd_ready": bool(svc and svc._svd is not None),
        "movies": len(svc.movies) if svc else 0,
        "indian_movies": indian,
        "embedding": model_info(),
    }


@app.get("/api/movies/search")
def search_movies(
    q: str = Query(..., min_length=1, description="Natural-language query"),
    n: int = Query(10, ge=1, le=50),
    origin: str | None = Query(None, description="Filter by catalog origin: movielens | indian"),
    language: str | None = Query(None, description="Filter by language, e.g. Hindi, Tamil"),
):
    """Semantic / natural-language search over the catalog."""
    svc = _require_service()
    try:
        results = svc.semantic_search(q, n=n, origin=origin, language=language)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"query": q, "origin": origin, "language": language, "results": results}


@app.get("/api/movies/trending")
def trending_movies(
    n: int = Query(12, ge=1, le=50),
    origin: str | None = Query(None, description="Filter by catalog origin: movielens | indian"),
    language: str | None = Query(None, description="Filter by language, e.g. Hindi, Tamil"),
):
    """Top movies by popularity (Bayesian average over ratings)."""
    svc = _require_service()
    return {
        "origin": origin,
        "language": language,
        "results": svc.trending(n=n, origin=origin, language=language),
    }


@app.get("/api/recommend/{user_id}")
def recommend_user(
    user_id: int,
    n: int = Query(10, ge=1, le=50),
    with_explanations: bool = Query(True),
):
    """Hybrid recommendations for a MovieLens user."""
    svc = _require_service()
    recs = svc.recommend(user_id, n=n, with_explanations=with_explanations)
    return {"user_id": user_id, "recommendations": recs}


@app.post("/api/recommend/query")
def recommend_query(req: RecommendQueryRequest):
    """Recommendations blended with a natural-language preference query."""
    svc = _require_service()
    recs = svc.recommend(
        req.user_id,
        n=req.n,
        query=req.query,
        with_explanations=req.with_explanations,
    )
    return {"user_id": req.user_id, "query": req.query, "recommendations": recs}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    """Record a real-time user signal (like/dislike/watchlist/remove)."""
    svc = _require_service()
    svc.feedback.record(req.user_id, req.movie_id, req.action)
    return {"status": "recorded", "user_id": req.user_id, "action": req.action}


@app.get("/api/user/{user_id}/profile")
def user_profile(user_id: int, min_rating: float = Query(4.0)):
    svc = _require_service()
    ratings = svc.user_ratings(user_id, min_rating=min_rating)
    return {
        "user_id": user_id,
        "has_history": bool(ratings),
        "liked": ratings,
        "watchlist": sorted(svc.feedback.watchlist(user_id)),
        "liked_feedback": sorted(svc.feedback.liked(user_id)),
    }


@app.get("/api/movies/{movie_id}")
def movie(movie_id: int):
    svc = _require_service()
    item = svc.get_movie(movie_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return item


@app.get("/api/explain/{user_id}/{movie_id}")
def explain_movie(user_id: int, movie_id: int):
    """Explain why a single movie is a good match for a user."""
    svc = _require_service()
    item = svc.get_movie(movie_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    liked = svc.liked_movies(user_id)
    from cinematch import explainer

    return {"user_id": user_id, "movie": item, "why": explainer.explain(item, liked)}
