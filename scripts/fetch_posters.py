"""Fetch TMDB poster URLs (and optionally plot overviews) for the catalog.

Posters are cached to ``data/processed/posters.parquet`` and written back into
``data/processed/movies.parquet``; plot overviews go to
``data/processed/overviews.parquet`` and are picked up automatically by
``index_vectors.py`` for richer embeddings.

Requires a TMDB API key (https://www.themoviedb.org/settings/api) in the
``TMDB_API_KEY`` environment variable (see ``.env.example``).

    python scripts/fetch_posters.py [--limit N] [--sleep 0.25] [--fill-overviews]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from cinematch.config import SETTINGS
from cinematch.data import load_processed

POSTER_BASE = "https://image.tmdb.org/t/p/w342"
_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"


def _best_match(results: list[dict], title: str, year: int | None) -> dict | None:
    """Pick the TMDB result that best matches title + year."""
    best, best_score = None, -1.0
    for r in results:
        score = 0.0
        if r.get("title", "").strip().lower() == title.strip().lower():
            score += 3.0
        release = r.get("release_date") or ""
        result_year = int(release[:4]) if len(release) >= 4 and release[:4].isdigit() else None
        if year is not None and result_year is not None:
            score += 2.0 - min(abs(result_year - year), 5) * 0.4
        score += r.get("popularity", 0.0) / 1000.0
        if score > best_score:
            best, best_score = r, score
    return best


def _load_map(path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out = {}
    for row in df.itertuples(index=False):
        out[int(row.movie_id)] = "" if pd.isna(row.value) else str(row.value)
    return out


def _save_map(path, mapping: dict[int, str]) -> None:
    frame = pd.DataFrame(
        [{"movie_id": mid, "value": val} for mid, val in mapping.items()]
    ).sort_values("movie_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TMDB posters (and overviews) for the catalog.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N movies (testing)")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between TMDB calls (rate limit)")
    parser.add_argument("--key", default=None, help="Override TMDB API key")
    parser.add_argument("--fill-overviews", action="store_true", help="Backfill plot overviews for all movies")
    args = parser.parse_args()

    key = args.key or SETTINGS.tmdb_api_key
    if not key:
        print("No TMDB API key set. Add TMDB_API_KEY to .env or pass --key.")
        return

    processed_dir = SETTINGS.paths.processed_dir
    data = load_processed(paths=SETTINGS.paths)
    movies = data["movies"].copy()

    poster_path = processed_dir / "posters.parquet"
    overview_path = processed_dir / "overviews.parquet"
    posters = _load_map(poster_path)
    overviews = _load_map(overview_path)

    meta = {
        int(row.movie_id): (str(row.clean_title), None if pd.isna(row.year) else int(row.year))
        for row in movies.itertuples(index=False)
    }

    def fetch(movie_id: int) -> dict | None:
        title, year = meta[movie_id]
        try:
            resp = requests.get(
                _SEARCH_URL,
                params={
                    "api_key": key,
                    "query": title,
                    "include_adult": "false",
                    "language": "en-US",
                },
                timeout=10,
            )
            resp.raise_for_status()
            results = (resp.json().get("results") or [])[:8]
            return _best_match(results, title, year)
        except Exception as exc:
            print(f"  ! {movie_id} ({title}): {exc}")
            return None

    session_progress = {"count": 0, "t0": time.time()}

    def tick(total: int, label: str) -> None:
        session_progress["count"] += 1
        if session_progress["count"] % 50 == 0:
            elapsed = time.time() - session_progress["t0"]
            print(f"  {label}: {session_progress['count']}/{total} in {elapsed:.0f}s")

    todo = [mid for mid in meta if mid not in posters]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Fetching posters for {len(todo):,} movies ({len(posters):,} already cached) ...")
    for movie_id in todo:
        hit = fetch(movie_id)
        if hit is not None and hit.get("poster_path"):
            posters[movie_id] = f"{POSTER_BASE}{hit['poster_path']}"
        else:
            posters.setdefault(movie_id, "")
        if hit is not None and hit.get("overview"):
            overviews.setdefault(movie_id, str(hit["overview"]))
        tick(len(todo), "posters")
        time.sleep(args.sleep)
    _save_map(poster_path, posters)
    _save_map(overview_path, overviews)

    if args.fill_overviews:
        missing = [mid for mid in meta if mid not in overviews or not overviews[mid]]
        if args.limit is not None:
            missing = missing[: args.limit]
        print(f"Backfilling overviews for {len(missing):,} movies ...")
        for movie_id in missing:
            hit = fetch(movie_id)
            if hit is not None and hit.get("overview"):
                overviews[movie_id] = str(hit["overview"])
            else:
                overviews.setdefault(movie_id, "")
            tick(len(missing), "overviews")
            time.sleep(args.sleep)
        _save_map(overview_path, overviews)

    movies["poster_url"] = movies["movie_id"].map(posters).fillna("")
    movies.to_parquet(processed_dir / "movies.parquet")

    total = len(meta)
    has = sum(1 for u in posters.values() if u)
    ov = sum(1 for u in overviews.values() if u)
    print(f"Done. {has:,}/{total:,} movies have posters ({100.0 * has / total:.0f}%) | "
          f"{ov:,} overviews cached.")
    print("Re-run `python scripts/index_vectors.py --reset` to embed posters into the vector index.")


if __name__ == "__main__":
    main()
