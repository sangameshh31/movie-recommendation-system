"""Data ingestion: MovieLens download, parsing and caching."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from cinematch.config import DataConfig, Paths

# Genres present in the MovieLens 100k/1M u.item files (in column order).
_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

_BASE_URL = "https://files.grouplens.org/datasets/movielens"

_SIZE_DIRS = {"100k": "ml-100k", "1m": "ml-1m"}
_SIZE_FILES = {"100k": "ml-100k.zip", "1m": "ml-1m.zip"}


def _url_for(size: str) -> str:
    return f"{_BASE_URL}/{_SIZE_FILES[size]}"


def download(size: str | None = None, paths: Paths | None = None) -> Path:
    """Download and extract the MovieLens dataset if not already present.

    Returns the directory containing the extracted dataset.
    """
    cfg = DataConfig() if size is None else DataConfig(size=size)
    paths = paths or Paths()
    dir_name = _SIZE_DIRS[cfg.size]
    dest_dir = paths.raw_dir / dir_name

    if dest_dir.exists():
        return dest_dir

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = paths.raw_dir / _SIZE_FILES[cfg.size]
    if not zip_path.exists():
        print(f"Downloading MovieLens {cfg.size} from {_url_for(cfg.size)} ...")
        urlretrieve(_url_for(cfg.size), zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(paths.raw_dir)

    return dest_dir


def load_movies(data_dir: Path) -> pd.DataFrame:
    """Load movie metadata (title, year, genres) from the raw dataset."""
    if data_dir.name == "ml-100k":
        file = data_dir / "u.item"
        if not file.exists():
            raise FileNotFoundError(f"Missing {file}. Run `python scripts/download_data.py`.")
        cols = ["movie_id", "title", "release_date", "video_release", "imdb_url", *_GENRES]
        movies = pd.read_csv(file, sep="|", encoding="latin-1", names=cols)
        genres = movies[_GENRES].astype(bool)
        movies["genres"] = [
            [g for g, keep in zip(_GENRES, row) if keep]
            for row in genres.itertuples(index=False)
        ]
        movies = movies[["movie_id", "title", "genres"]]
    else:  # ml-1m
        file = data_dir / "movies.dat"
        if not file.exists():
            raise FileNotFoundError(f"Missing {file}. Run `python scripts/download_data.py`.")
        movies = pd.read_csv(
            file, sep="::", engine="python", encoding="latin-1",
            names=["movie_id", "title", "genres_str"],
        )
        movies["genres"] = movies["genres_str"].str.split("|")
        movies = movies[["movie_id", "title", "genres"]]

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return movies


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load ratings (user_id, movie_id, rating, timestamp)."""
    if data_dir.name == "ml-100k":
        file = data_dir / "u.data"
        cols = ["user_id", "movie_id", "rating", "timestamp"]
    else:  # ml-1m
        file = data_dir / "ratings.dat"
        cols = ["user_id", "movie_id", "rating", "timestamp"]
    if not file.exists():
        raise FileNotFoundError(f"Missing {file}. Run `python scripts/download_data.py`.")
    sep = "\t" if data_dir.name == "ml-100k" else "::"
    return pd.read_csv(file, sep=sep, engine="python", names=cols)


def ingest(paths: Paths | None = None, size: str | None = None) -> dict[str, pd.DataFrame]:
    """Run the full ETL and cache normalized frames to parquet.

    Returns {"movies": ..., "ratings": ...}.
    """
    paths = paths or Paths()
    cfg = DataConfig() if size is None else DataConfig(size=size)
    data_dir = download(size=cfg.size, paths=paths)

    movies = load_movies(data_dir)
    ratings = load_ratings(data_dir)

    # Keep only movies that actually have ratings (drops dangling metadata rows).
    rated_ids = set(ratings["movie_id"].unique())
    movies = movies[movies["movie_id"].isin(rated_ids)].reset_index(drop=True)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(paths.processed_dir / "movies.parquet")
    ratings.to_parquet(paths.processed_dir / "ratings.parquet")

    print(f"Ingested {len(movies):,} movies, {len(ratings):,} ratings from MovieLens {cfg.size.upper()}.")
    return {"movies": movies, "ratings": ratings}


def load_processed(paths: Paths | None = None) -> dict[str, pd.DataFrame]:
    """Load cached parquet frames (calling :func:`ingest` if cache is absent).

    Applies the Indian cinema augmentation so every consumer (train, eval,
    index, API) sees the combined catalog.
    """
    paths = paths or Paths()
    movies_path = paths.processed_dir / "movies.parquet"
    ratings_path = paths.processed_dir / "ratings.parquet"
    if not (movies_path.exists() and ratings_path.exists()):
        data = ingest(paths=paths)
        movies, ratings = data["movies"], data["ratings"]
    else:
        movies = pd.read_parquet(movies_path)
        ratings = pd.read_parquet(ratings_path)

    from cinematch.indian_cinema import apply_indian_augmentation

    movies, ratings = apply_indian_augmentation(movies, ratings, paths.processed_dir)

    from cinematch.tmdb_catalog import apply_tmdb_augmentation

    movies, ratings = apply_tmdb_augmentation(movies, ratings, paths.processed_dir)
    return {"movies": movies, "ratings": ratings}
