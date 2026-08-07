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
        ...,
        description="One of: like | dislike | watchlist | watched | rate | remove",
    )
    value: float | None = Field(
        None, ge=0.5, le=5.0, description="Star rating (0.5-5) when action='rate'"
    )


class RefineRequest(BaseModel):
    seed: str = Field(..., min_length=1, description="Original search query")
    additions: list[str] = Field(default_factory=list, max_length=5)
    removals: list[str] = Field(default_factory=list, max_length=5)
    n: int = Field(12, ge=1, le=50)
    origin: str | None = None
    language: str | None = None


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
    indian = series = anime = 0
    if svc is not None and "origin" in svc.movies.columns:
        indian = int((svc.movies["origin"] == "indian").sum())
        series = int((svc.movies["origin"] == "series").sum())
        anime = int((svc.movies["origin"] == "anime").sum())
    return {
        "status": "ok" if svc is not None else "loading",
        "vector_index": bool(svc and svc.vector_index_ready),
        "svd_ready": bool(svc and svc._svd is not None),
        "movies": len(svc.movies) if svc else 0,
        "indian_movies": indian,
        "series": series,
        "anime": anime,
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
    origin: str | None = Query(None, description="Filter by catalog origin: movielens | indian | anime | series | new"),
    language: str | None = Query(None, description="Filter by language, e.g. Hindi, Tamil"),
    media_type: str | None = Query(None, description="Filter by media type: movie | series"),
    sort_by: str = Query("popularity", description="Sort key: popularity | rating | new"),
    genre: str | None = Query(None, description="Require a genre, e.g. Comedy, Sci-Fi"),
):
    """Top titles by popularity / rating / recency, optionally filtered."""
    svc = _require_service()
    return {
        "origin": origin,
        "language": language,
        "media_type": media_type,
        "sort_by": sort_by,
        "genre": genre,
        "results": svc.trending(
            n=n,
            origin=origin,
            language=language,
            media_type=media_type,
            sort_by=sort_by,
            genre=genre,
        ),
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
    """Record a real-time user signal (like/dislike/watchlist/watched/rate)."""
    svc = _require_service()
    svc.feedback.record(req.user_id, req.movie_id, req.action, value=req.value)
    return {"status": "recorded", "user_id": req.user_id, "action": req.action}


@app.post("/api/search/refine")
def refine_search(req: RefineRequest):
    """Conversational refinement: start from a seed query, then nudge the
    vector with additional concepts (``+more like X``) and subtract others
    (``-less of Y``) before re-searching the catalog."""
    from cinematch.embeddings import embed_query

    svc = _require_service()
    vec = embed_query(req.seed.strip())
    nudge = None
    for phrase in req.additions:
        if phrase.strip():
            nudge = embed_query(phrase.strip()) if nudge is None else nudge + embed_query(phrase.strip())
    for phrase in req.removals:
        if phrase.strip():
            nudge = -embed_query(phrase.strip()) if nudge is None else nudge - embed_query(phrase.strip())
    if nudge is not None:
        vec = vec + 0.6 * nudge
    results = svc.semantic_search_vector(vec, n=req.n, origin=req.origin, language=req.language)
    return {
        "seed": req.seed,
        "additions": req.additions,
        "removals": req.removals,
        "results": results,
    }


@app.get("/api/surprise")
def surprise(user_id: int = Query(..., description="User id")):
    """A single personalized pick with an element of randomness."""
    svc = _require_service()
    pick = svc.surprise(user_id)
    if pick is None:
        raise HTTPException(status_code=404, detail="No picks available")
    return {"user_id": user_id, "pick": pick}


@app.get("/api/people/search")
def people_search(
    q: str = Query(..., min_length=1, description="Director/actor name"),
    n: int = Query(24, ge=1, le=50),
):
    """Resolve a person and their credits that exist in the catalog."""
    from cinematch.people import PeopleService

    svc = _require_service()
    people = svc.people
    person = people.search(q, svc.get_movie)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    person["catalog_movies"] = person["catalog_movies"][:n]
    return person


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


@app.get("/api/user/{user_id}/library")
def user_library(user_id: int):
    """The user's persistent account library (likes, stars, watchlist, watched)."""
    svc = _require_service()
    profile = svc.feedback.profile(user_id)
    return {
        "user_id": user_id,
        "liked": svc.get_movies(profile["liked"]),
        "disliked": svc.get_movies(profile["disliked"]),
        "watchlist": svc.get_movies(profile["watchlist"]),
        "watched": svc.get_movies(profile["watched"]),
        "stars": {
            str(mid): value
            for mid, value in sorted(profile["stars"].items(), key=lambda kv: kv[1], reverse=True)
        },
    }


@app.get("/api/movies/{movie_id}")
def movie(movie_id: int):
    svc = _require_service()
    item = svc.get_movie(movie_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return item


@app.get("/api/movies/{movie_id}/details")
def movie_details(movie_id: int, similar: int = Query(10, ge=0, le=24)):
    """Full detail (plot, cast, director, producer) plus similar titles."""
    svc = _require_service()
    item = svc.get_movie(movie_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    try:
        details = svc.details.enrich(movie_id)
    except Exception as exc:  # pragma: no cover
        from cinematch.details import _empty_payload

        print(f"Details enrichment failed for {movie_id}: {exc}")
        details = _empty_payload("error")
    similar_movies = svc.similar_movies(movie_id, n=similar) if similar else []
    return {"movie": item, "details": details, "similar": similar_movies}


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
