"""Fetch a large catalog from TMDB.

Sources (all captured from the same responses, so no extra poster/overview
passes are needed):

* **Popular movies** — the most popular films on TMDB (all-time).
* **Top rated movies** — the highest-rated films (vote_count-gated).
* **Recent releases** — the most popular films of 2024, 2025 and 2026.
* **Classic Indian cinema** — top-voted films of 13 Indian languages,
  including the full Dr. Rajkumar filmography.
* **Anime** — Japanese animated movies *and* TV series.
* **Animation / cartoons** — the Animation genre across languages.
* **TV series** — the most popular and highest-rated TV shows.

Every row carries ``media_type`` (``movie`` / ``series``) and ``origin``
(``indian`` / ``anime`` / ``series`` / ``new``) so the UI can render
separate rails. Results are deduped against the existing catalog and cached
to ``data/processed/tmdb_catalog.parquet``. Rerun any time to pull more
(each run only adds movies not already present, so it is safe and resumable;
the cache is flushed to disk periodically).

    python scripts/fetch_catalog.py
    python scripts/fetch_catalog.py --popular-pages 200 --classics 300
"""

from __future__ import annotations

import argparse
import re
import time

import pandas as pd
import requests

from cinematch.config import SETTINGS
from cinematch.data import load_processed
from cinematch.tmdb_net import patch_tmdb_dns

POSTER_BASE = "https://image.tmdb.org/t/p/w342"
_API = "https://api.themoviedb.org/3"

INDIAN_LANGS = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "as": "Assamese",
    "or": "Oriya",
    "ur": "Urdu",
    "bho": "Bhojpuri",
}
RELEASE_YEARS = [2024, 2025, 2026]
MIN_VOTES_BY_YEAR = {2024: 40, 2025: 25, 2026: 10}

_session = requests.Session()


def _get(key: str, path: str, params: dict, tries: int = 4) -> dict:
    last: Exception | None = None
    for _ in range(tries):
        try:
            resp = _session.get(
                f"{_API}{path}", params={"api_key": key, **params}, timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.0)
    raise last  # type: ignore[misc]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _find_rajkumar(key: str) -> int | None:
    """Return the TMDB person id of the Kannada legend Dr. Rajkumar."""
    queries = ["Dr. Rajkumar", "Rajkumar", "Singanalluru"]
    best: tuple[int, int] | None = None  # (person_id, kannada_credit_count)
    for query in queries:
        people = _get(key, "/search/person", {"query": query}).get("results") or []
        for person in people:
            name = (person.get("name") or "").strip().lower()
            if "rajkumar" not in name:
                continue
            if (person.get("known_for_department") or "") not in ("Acting", ""):
                continue
            credits = _get(key, f"/person/{person['id']}/movie_credits", {})
            kn = sum(1 for c in credits.get("cast", []) if c.get("original_language") == "kn")
            if kn >= 8 and (best is None or kn > best[1]):
                best = (person["id"], kn)
        time.sleep(0.2)
    if best is not None:
        print(f"  Dr. Rajkumar -> person id {best[0]} with {best[1]} Kannada credits")
        return best[0]
    return None


class Collector:
    """Holds rows, dedupes and periodically persists the resume cache."""

    def __init__(self, out_path, movies, args):
        self.out_path = out_path
        self.seen = {
            (_norm(row.clean_title), None if pd.isna(row.year) else int(row.year))
            for row in movies.itertuples(index=False)
        }
        self.seen_ids = (
            {int(x) for x in movies["tmdb_id"].dropna().unique()}
            if "tmdb_id" in movies.columns
            else set()
        )
        self.rows: list[dict] = []
        if out_path.exists():
            prev = pd.read_parquet(out_path)
            self.rows = prev.to_dict("records")
            for r in self.rows:
                self.seen_ids.add(int(r["tmdb_id"]))
        self.new_rows: list[dict] = []
        self.skipped = 0
        self.args = args
        self._since_save = 0

    def _flush(self) -> None:
        if not self.rows:
            return
        frame = pd.DataFrame(self.rows).drop_duplicates(subset=["tmdb_id"])
        cols = [
            "tmdb_id", "clean_title", "year", "genres", "language", "origin",
            "media_type", "vote_average", "vote_count", "poster_url", "overview",
        ]
        for col in cols:
            if col not in frame.columns:
                frame[col] = "" if col not in ("vote_average", "vote_count") else 0
        frame = frame[cols].sort_values(["year", "tmdb_id"]).reset_index(drop=True)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.out_path)

    def add(self, item: dict, origin: str, language: str, media_type: str) -> bool:
        mid = int(item.get("id"))
        if mid in self.seen_ids:
            self.skipped += 1
            return False
        title = (item.get("title") or item.get("name") or "").strip()
        release = item.get("release_date") or item.get("first_air_date") or ""
        year = int(release[:4]) if len(release) >= 4 and release[:4].isdigit() else None
        if not title or year is None:
            self.skipped += 1
            return False
        key_t = (_norm(title), year)
        if key_t in self.seen or any(
            r["year"] == year and _norm(r["clean_title"]) == key_t[0] for r in self.new_rows
        ):
            self.skipped += 1
            return False
        if origin is None:
            lang_iso = item.get("original_language") or ""
            language = INDIAN_LANGS.get(lang_iso, "") or ""
            origin = "indian" if language else "new"
        poster = item.get("poster_path")
        row = {
            "tmdb_id": mid,
            "clean_title": title,
            "year": year,
            "genres": [
                genre_map[gid] for gid in (item.get("genre_ids") or []) if gid in genre_map
            ],
            "language": language,
            "origin": origin,
            "media_type": media_type,
            "vote_average": float(item.get("vote_average") or 0.0),
            "vote_count": int(item.get("vote_count") or 0),
            "poster_url": f"{POSTER_BASE}{poster}" if poster else "",
            "overview": (item.get("overview") or "").strip(),
        }
        self.rows.append(row)
        self.new_rows.append(row)
        self.seen_ids.add(mid)
        self.seen.add(key_t)
        self._since_save += 1
        if self._since_save >= 200:
            self._flush()
            self._since_save = 0
        return True


genre_map: dict[int, str] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a large TMDB catalog.")
    parser.add_argument("--popular-pages", type=int, default=150, help="Pages (x20) of all-time popular movies")
    parser.add_argument("--top-pages", type=int, default=50, help="Pages (x20) of top-rated movies")
    parser.add_argument("--releases", type=int, default=250, help="Top-N popular movies per year for 2024-2026")
    parser.add_argument("--classics", type=int, default=200, help="Top-N classic Indian films per language")
    parser.add_argument("--classic-min-votes", type=int, default=20, help="Min vote_count for Indian-language discover")
    parser.add_argument("--anime", type=int, default=600, help="Top-N Japanese animated movies")
    parser.add_argument("--anime-series", type=int, default=400, help="Top-N Japanese animated TV series")
    parser.add_argument("--animation-pages", type=int, default=20, help="Pages (x20) of Animation-genre movies")
    parser.add_argument("--series-pages", type=int, default=25, help="Pages (x20) per TV series sort")
    parser.add_argument("--no-rajkumar", action="store_true", help="Skip the Dr. Rajkumar filmography pass")
    parser.add_argument("--sleep", type=float, default=0.15, help="Minimum seconds between API calls")
    parser.add_argument("--key", default=None, help="Override TMDB API key")
    args = parser.parse_args()

    key = args.key or SETTINGS.tmdb_api_key
    if not key:
        print("No TMDB API key set. Add TMDB_API_KEY to .env or pass --key.")
        return

    patch_tmdb_dns()

    processed_dir = SETTINGS.paths.processed_dir
    out_path = processed_dir / "tmdb_catalog.parquet"

    movies = load_processed(paths=SETTINGS.paths)["movies"]
    col = Collector(out_path, movies, args)

    genre_map.clear()
    try:
        for g in _get(key, "/genre/movie/list", {})["genres"]:
            genre_map[int(g["id"])] = g["name"]
        for g in _get(key, "/genre/tv/list", {})["genres"]:
            genre_map[int(g["id"])] = g["name"]
    except requests.RequestException:
        pass

    def paged_movie(params: dict, cap: int, max_pages: int, origin: str | None, language: str) -> None:
        page, got = 1, 0
        while got < cap and page <= max_pages:
            try:
                items = _get(key, "/discover/movie", {**params, "page": page})["results"]
            except requests.RequestException as exc:
                print(f"  ! discover/movie failed: {exc}")
                break
            if not items:
                break
            for item in items:
                if col.add(item, origin, language, "movie"):
                    got += 1
            page += 1
            time.sleep(args.sleep)

    def paged_tv(params: dict, cap: int, max_pages: int, origin: str | None, language: str) -> None:
        page, got = 1, 0
        while got < cap and page <= max_pages:
            try:
                items = _get(key, "/discover/tv", {**params, "page": page})["results"]
            except requests.RequestException as exc:
                print(f"  ! discover/tv failed: {exc}")
                break
            if not items:
                break
            for item in items:
                if col.add(item, origin, language, "series"):
                    got += 1
            page += 1
            time.sleep(args.sleep)

    def section_added() -> int:
        return len(col.new_rows)

    t0 = time.time()

    # 1) All-time popular movies
    before = section_added()
    paged_movie(
        {"sort_by": "popularity.desc", "vote_count.gte": 20},
        args.popular_pages * 20,
        args.popular_pages,
        None,
        "",
    )
    print(f"  popular movies: +{section_added() - before}")

    # 2) Top rated movies
    before = section_added()
    paged_movie(
        {"sort_by": "vote_average.desc", "vote_count.gte": 1000},
        args.top_pages * 20,
        args.top_pages,
        None,
        "",
    )
    print(f"  top rated movies: +{section_added() - before}")

    # 3) Recent releases (2024-2026)
    today = time.strftime("%Y-%m-%d")
    for year in RELEASE_YEARS:
        before = section_added()
        paged_movie(
            {
                "sort_by": "popularity.desc",
                "primary_release_date.gte": f"{year}-01-01",
                "primary_release_date.lte": today if year == 2026 else f"{year}-12-31",
                "vote_count.gte": MIN_VOTES_BY_YEAR.get(year, 10),
            },
            args.releases,
            15,
            None,
            "",
        )
        print(f"  releases {year}: +{section_added() - before}")

    # 4) Classic Indian films per language
    for iso, name in INDIAN_LANGS.items():
        before = section_added()
        paged_movie(
            {
                "sort_by": "vote_count.desc",
                "with_original_language": iso,
                "vote_count.gte": args.classic_min_votes,
            },
            args.classics,
            25,
            "indian",
            name,
        )
        print(f"  classics {name}: +{section_added() - before}")

    # 5) Dr. Rajkumar filmography
    if not args.no_rajkumar:
        before = section_added()
        pid = _find_rajkumar(key)
        if pid is not None:
            try:
                credits = _get(key, f"/person/{pid}/movie_credits", {})
            except requests.RequestException as exc:
                print(f"  ! rajkumar credits failed: {exc}")
                credits = {}
            cast = sorted(
                credits.get("cast") or [],
                key=lambda c: -(c.get("vote_count") or 0),
            )
            for credit in cast[:200]:
                lang = INDIAN_LANGS.get(credit.get("original_language") or "", "")
                if lang:
                    col.add(credit, "indian", lang, "movie")
            print(f"  rajkumar filmography: +{section_added() - before}")

    # 6) Anime movies (Japanese animation)
    before = section_added()
    paged_movie(
        {"sort_by": "vote_count.desc", "with_original_language": "ja", "with_genres": "16"},
        args.anime,
        args.anime // 20 + 2,
        "anime",
        "Japanese",
    )
    print(f"  anime movies: +{section_added() - before}")

    # 7) Anime TV series (Japanese animation)
    before = section_added()
    paged_tv(
        {"sort_by": "vote_count.desc", "with_original_language": "ja", "with_genres": "16"},
        args.anime_series,
        args.anime_series // 20 + 2,
        "anime",
        "Japanese",
    )
    print(f"  anime series: +{section_added() - before}")

    # 8) Animation / cartoons (all languages)
    before = section_added()
    paged_movie(
        {"sort_by": "popularity.desc", "with_genres": "16", "vote_count.gte": 20},
        args.animation_pages * 20,
        args.animation_pages,
        None,
        "",
    )
    print(f"  animation movies: +{section_added() - before}")

    # 9) TV series (top rated + popular)
    before = section_added()
    paged_tv(
        {"sort_by": "vote_average.desc", "vote_count.gte": 500},
        args.series_pages * 20,
        args.series_pages,
        "series",
        "",
    )
    paged_tv(
        {"sort_by": "popularity.desc", "vote_count.gte": 20},
        args.series_pages * 20,
        args.series_pages,
        "series",
        "",
    )
    print(f"  tv series: +{section_added() - before}")

    col._flush()
    if col.new_rows:
        print(f"Wrote {len(col.rows):,} movies to {out_path.name} "
              f"in {time.time() - t0:.0f}s ({len(col.new_rows):,} new this run, "
              f"{col.skipped:,} skipped as already present)")
    else:
        print(f"No new movies to add ({col.skipped:,} already present).")

    print("Next: `python scripts/migrate_catalog.py` then `python scripts/train.py --force` "
          "&& `python scripts/index_vectors.py --reset`, then restart the API.")


if __name__ == "__main__":
    main()
