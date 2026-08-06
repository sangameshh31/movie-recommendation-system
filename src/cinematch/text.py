"""Movie metadata -> rich text used for semantic embeddings.

MovieLens ships minimal metadata (title + genres), so we build a compact
descriptor from what we have. When a TMDB API key is configured we can
upgrade to real plot overviews via :func:`fetch_tmdb_overviews`.
"""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd

from cinematch.config import SETTINGS

_OverviewFetcher = Callable[[int], str | None]


def build_movie_text(movies: pd.DataFrame, overview_fetcher: _OverviewFetcher | None = None) -> pd.DataFrame:
    """Return a copy of ``movies`` with a ``text`` column for embedding.

    Text layout: ``title (year) :: Genre1, Genre2 ...`` plus an optional TMDB
    plot overview when available.
    """
    df = movies.copy()
    genre_str = df["genres"].apply(lambda g: ", ".join(g) if isinstance(g, list) else str(g))
    df["text"] = df.apply(
        lambda row: _compose(row, genre_str.loc[row.name], overview_fetcher), axis=1
    )
    return df


def _compose(row: pd.Series, genre_str: str, overview_fetcher: _OverviewFetcher | None) -> str:
    year = row.get("year")
    year_str = f" ({int(year)})" if pd.notna(year) else ""
    base = f"{row['clean_title']}{year_str} :: {genre_str}"

    if overview_fetcher is not None:
        overview = overview_fetcher(int(row["movie_id"]))
        if overview:
            return f"{base} :: {overview}"
    return base


def build_query_text(query: str, genres: list[str] | None = None) -> str:
    """Wrap a raw natural-language query so it matches the embedding space."""
    if genres:
        return f"{query} :: {', '.join(genres)}"
    return query


# ---------------------------------------------------------------------------
# Optional TMDB enrichment (used only when TMDB_API_KEY is set)
# ---------------------------------------------------------------------------

def make_tmdb_fetcher(movies: pd.DataFrame, api_key: str = SETTINGS.tmdb_api_key):
    """Return a ``movie_id -> overview`` fetcher backed by TMDB v3 search.

    Returns ``None`` when no API key is supplied. Rate-limits politely.
    """
    if not api_key:
        return None

    import requests

    titles = {int(row.movie_id): row.clean_title for row in movies.itertuples(index=False)}
    _cache: dict[int, str | None] = {}
    _timestamps: list[float] = []

    def _rate_limit() -> None:
        now = time.time()
        _timestamps[:] = [t for t in _timestamps if now - t < 60.0]
        if len(_timestamps) >= max_requests_per_min:
            time.sleep(60.0 - (now - _timestamps[0]) + 0.5)
            _timestamps[:] = []
        _timestamps.append(time.time())

    def fetch(movie_id: int) -> str | None:
        if movie_id in _cache:
            return _cache[movie_id]
        title = titles.get(movie_id)
        if title is None:
            return None
        _rate_limit()
        try:
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": api_key, "query": title, "language": "en-US"},
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
            overview = results[0].get("overview") if results else None
        except Exception:
            overview = None
        _cache[movie_id] = overview
        return overview

    return fetch
