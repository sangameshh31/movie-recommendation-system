"""LLM-powered explanations ("Why you might like this").

Uses Ollama (local Llama) when available and silently falls back to a
template-based explainer otherwise, so the API never depends on an external
service being up.
"""

from __future__ import annotations

from functools import lru_cache

import httpx

from cinematch.config import SETTINGS

_TEMPLATE = (
    "You like {liked} and this shares the {shared} vibe, so it fits "
    "your taste for {genres}."
)


@lru_cache(maxsize=1)
def _ollama_available() -> bool:
    cfg = SETTINGS.explainer
    try:
        resp = httpx.get(f"{cfg.ollama_url.rstrip('/')}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ollama_generate(prompt: str) -> str | None:
    cfg = SETTINGS.explainer
    try:
        resp = httpx.post(
            f"{cfg.ollama_url.rstrip('/')}/api/generate",
            json={"model": cfg.ollama_model, "prompt": prompt, "stream": False},
            timeout=cfg.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def _template_explanation(candidate: dict, liked_movies: list[dict]) -> str:
    genres = candidate.get("genres", [])
    genre_text = ", ".join(genres) if genres else "a wide range of movies"
    if liked_movies:
        liked_titles = ", ".join(m["title"] for m in liked_movies[:3])
        liked_genres = {g for m in liked_movies for g in m.get("genres", [])}
        shared = [g for g in genres if g in liked_genres]
        shared_text = ", ".join(shared) if shared else "similar storylines and tone"
        return _TEMPLATE.format(
            liked=liked_titles, shared=shared_text, genres=genre_text
        )
    return (
        f"'{candidate.get('title')}' is a strong match for your profile — "
        f"a {genre_text.lower()} pick that balances critical acclaim and popularity."
    )


def explain(
    candidate: dict,
    liked_movies: list[dict] | None = None,
    use_llm: bool = True,
) -> dict:
    """Return a "why" explanation. Keys: ``text``, ``source`` (llm|template)."""
    liked_movies = liked_movies or []
    if use_llm and _ollama_available():
        liked_desc = (
            "; ".join(
                f"{m['title']} ({', '.join(m.get('genres', []))})" for m in liked_movies[:3]
            )
            or "no prior viewing history"
        )
        prompt = (
            "You are a movie recommendation assistant. In ONE sentence, explain "
            "why the movie is a good fit, referencing specific shared elements "
            "(genre, director, themes, tone). Keep it warm and specific.\n\n"
            f"Liked movies: {liked_desc}\n"
            f"Candidate: {candidate.get('title')} ({', '.join(candidate.get('genres', []))})\n\n"
            "Explanation:"
        )
        text = _ollama_generate(prompt)
        if text:
            return {"text": text, "source": "llm"}

    return {"text": _template_explanation(candidate, liked_movies), "source": "template"}
