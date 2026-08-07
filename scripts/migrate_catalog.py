"""One-off migration: add ``media_type`` + ``vote_average`` to the catalog.

* Every row gets ``media_type`` (``series`` for ``origin == "series"``,
  otherwise ``movie``).
* ``vote_average`` (0-10, IMDb-style badge) is backfilled from
  ``tmdb_catalog.parquet`` for TMDB titles, and from ``avg(rating) * 2`` for
  MovieLens titles.
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from cinematch.config import SETTINGS

processed = SETTINGS.paths.processed_dir
movies_path = processed / "movies.parquet"
ratings_path = processed / "ratings.parquet"
movies = pd.read_parquet(movies_path)
ratings = pd.read_parquet(ratings_path)

if "media_type" not in movies.columns:
    movies["media_type"] = np.where(movies["origin"] == "series", "series", "movie")
else:
    movies["media_type"] = movies["media_type"].where(
        movies["media_type"].notna() & (movies["media_type"].astype(str).str.strip() != ""),
        np.where(movies["origin"] == "series", "series", "movie"),
    )

if "vote_average" not in movies.columns:
    movies["vote_average"] = np.nan
missing_va = movies["vote_average"].isna()
if missing_va.any():
    cat = pd.read_parquet(processed / "tmdb_catalog.parquet")
    va = {int(r.tmdb_id): float(r.vote_average) for r in cat.itertuples(index=False)}
    movies.loc[missing_va, "vote_average"] = movies.loc[missing_va, "tmdb_id"].map(va)
    still_missing = movies["vote_average"].isna()
    if still_missing.any():
        avg = ratings.groupby("movie_id")["rating"].mean() * 2.0
        movies.loc[still_missing, "vote_average"] = movies.loc[still_missing, "movie_id"].map(avg)
    movies["vote_average"] = movies["vote_average"].round(1)

movies.to_parquet(movies_path)
n = movies["vote_average"].notna().sum()
print(f"media_type + vote_average applied to {len(movies):,} movies "
      f"({n:,} with a rating badge).")
print(f"  series={int((movies['media_type'] == 'series').sum()):,} "
      f"movies={int((movies['media_type'] == 'movie').sum()):,}")
