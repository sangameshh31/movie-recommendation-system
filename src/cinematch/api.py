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
    return {
        "status": "ok" if svc is not None else "loading",
        "vector_index": bool(svc and svc.vector_index_ready),
        "svd_ready": bool(svc and svc._svd is not None),
        "movies": len(svc.movies) if svc else 0,
        "embedding": model_info(),
    }


@app.get("/api/movies/search")
def search_movies(
    q: str = Query(..., min_length=1, description="Natural-language query"),
    n: int = Query(10, ge=1, le=50),
):
    """Semantic / natural-language search over the catalog."""
    svc = _require_service()
    try:
        results = svc.semantic_search(q, n=n)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"query": q, "results": results}


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
