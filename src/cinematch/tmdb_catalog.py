"""TMDB-driven catalog augmentation.

Adds the movies fetched by ``scripts/fetch_catalog.py`` (cached in
``data/processed/tmdb_catalog.parquet``) to the catalog — recent releases plus
classic Indian films (including Dr. Rajkumar's Kannada filmography). Each movie
gets a poster URL and plot overview captured from the fetch pass, plus seeded
ratings from the same synthetic "critic" users so the collaborative paths can
score them.

Idempotent and incremental: only movies whose ``tmdb_id`` is not already in the
catalog are added, so the catalog can be grown by simply re-running
``fetch_catalog.py`` and rebuilding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cinematch.indian_cinema import N_SEED_USERS

_COLUMNS = [
    "movie_id",
    "title",
    "clean_title",
    "year",
    "genres",
    "language",
    "origin",
    "media_type",
    "tmdb_id",
    "poster_url",
    "vote_average",
]


def _critic_user_ids(ratings: pd.DataFrame) -> list[int]:
    """The synthetic critic users created by the Indian augmentation."""
    uids = sorted({int(u) for u in ratings["user_id"].unique()})
    n = min(N_SEED_USERS, len(uids))
    return uids[-n:] if n else []


def seed_movie_ratings(
    movie_ids: list[int],
    vote_average: dict[int, float],
    vote_count: dict[int, int],
    ratings: pd.DataFrame,
    rng: np.random.Generator = np.random.default_rng(11),
) -> pd.DataFrame:
    """Build seeded critic ratings for a set of movies.

    Quality comes from the TMDB vote average (scaled to 1-5); popularity comes
    from the vote count, which raises the chance a given critic rated the film —
    so genuinely popular titles dominate trending while obscure ones still get a
    modest collaborative signal. Deterministic per (user, movie).
    """
    if not movie_ids or not ratings.shape[0]:
        return pd.DataFrame(columns=["user_id", "movie_id", "rating", "timestamp"])

    critics = _critic_user_ids(ratings)
    max_votes = max((vote_count.get(m, 1) for m in movie_ids), default=1)
    log_max = 1.0 + max(0.0, np.log1p(max_votes))
    jitter = np.array([-1, 0, 0, 1])

    rows = []
    for uid in critics:
        # Probability each critic has seen a film scales with its popularity.
        prob = {m: min(0.9, 0.12 + 0.5 * np.log1p(vote_count.get(m, 1)) / log_max) for m in movie_ids}
        for mid in movie_ids:
            if rng.random() > prob[mid]:
                continue
            base = max(1, min(5, int(round(vote_average.get(mid, 7.0) / 2.0))))
            rating = max(1.0, min(5.0, float(base + jitter[rng.integers(0, len(jitter))])))
            rows.append((uid, mid, rating, 0))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def apply_tmdb_augmentation(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    processed_dir,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append the TMDB catalog + seeded ratings (idempotent, incremental).

    Returns updated ``(movies, ratings)`` and, when ``processed_dir`` is given,
    persists the augmented frames back to the parquet cache along with any new
    plot overviews.
    """
    movies = movies.copy()
    ratings = ratings.copy()

    if processed_dir is None:
        return movies, ratings
    catalog_path = processed_dir / "tmdb_catalog.parquet"
    if not catalog_path.exists():
        return movies, ratings
    cat = pd.read_parquet(catalog_path)
    if cat.empty or "tmdb_id" not in cat.columns:
        return movies, ratings

    existing = (
        {int(x) for x in movies["tmdb_id"].dropna().unique()}
        if "tmdb_id" in movies.columns
        else set()
    )
    todo = cat[~cat["tmdb_id"].isin(existing)]
    if todo.empty:
        return movies, ratings

    # Ensure the marker columns exist (they do after the Indian augmentation).
    if "origin" not in movies.columns:
        movies["origin"] = "movielens"
        movies["language"] = ""

    next_id = int(movies["movie_id"].max()) + 1
    rows = []
    vote_of: dict[int, float] = {}
    count_of: dict[int, int] = {}
    overview_of: dict[int, str] = {}
    for i, row in enumerate(todo.itertuples(index=False)):
        mid = next_id + i
        clean = str(row.clean_title)
        year = int(row.year)
        vote = getattr(row, "vote_average", None)
        vote_of[mid] = float(vote) if pd.notna(vote) else 7.0
        count = getattr(row, "vote_count", None)
        count_of[mid] = int(count) if pd.notna(count) else 0
        overview = getattr(row, "overview", "")
        overview_of[mid] = "" if pd.isna(overview) else str(overview)
        rows.append(
            {
                "movie_id": mid,
                "title": f"{clean} ({year})",
                "clean_title": clean,
                "year": year,
                "genres": list(row.genres),
                "language": str(row.language or ""),
                "origin": str(row.origin or "new"),
                "media_type": str(getattr(row, "media_type", "") or "movie"),
                "tmdb_id": int(row.tmdb_id),
                "poster_url": str(getattr(row, "poster_url", "") or ""),
                "vote_average": vote_of[mid],
            }
        )

    new_movies = pd.DataFrame(rows, columns=_COLUMNS)
    movies = pd.concat([movies, new_movies], ignore_index=True)

    # Seed ratings from the critic users so popularity + item-CF can score the
    # new titles; quality scales with the TMDB vote average, and popularity
    # (vote count) controls how many critics have "seen" each film.
    new_ids = new_movies["movie_id"].tolist()
    seed_ratings = seed_movie_ratings(new_ids, vote_of, count_of, ratings)
    ratings = pd.concat([ratings, seed_ratings], ignore_index=True)

    # Merge plot overviews into the shared overview cache for richer embeddings.
    overview_path = processed_dir / "overviews.parquet"
    if overview_path.exists():
        try:
            cache = pd.read_parquet(overview_path)
        except Exception:
            cache = pd.DataFrame(columns=["movie_id", "overview"])
        cached_ids = {int(r.movie_id) for r in cache.itertuples(index=False)}
        fresh = [
            {"movie_id": mid, "overview": text}
            for mid, text in overview_of.items()
            if mid not in cached_ids and text
        ]
        if fresh:
            pd.concat([cache, pd.DataFrame(fresh)], ignore_index=True).to_parquet(
                overview_path
            )

    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        movies.to_parquet(processed_dir / "movies.parquet")
        ratings.to_parquet(processed_dir / "ratings.parquet")

    n_indian = int((new_movies["origin"] == "indian").sum())
    n_new = int((new_movies["origin"] == "new").sum())
    n_series = int((new_movies["origin"] == "series").sum())
    n_anime = int((new_movies["origin"] == "anime").sum())
    print(
        f"TMDB catalog augmentation: added {len(new_movies):,} movies "
        f"({n_indian:,} Indian / {n_anime:,} anime / {n_series:,} series / "
        f"{n_new:,} new) + {len(seed_ratings):,} seeded ratings."
    )
    return movies, ratings
