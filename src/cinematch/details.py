"""On-demand movie/series details (plot, cast, director, producer) from TMDB.

The catalog stores only the fields captured by the discover endpoints, so full
detail — synopsis, tagline, runtime, cast and crew — is fetched lazily from the
TMDB ``/movie``, ``/tv`` and ``/search`` endpoints the first time a user opens
a movie and cached to ``data/processed/details_cache.parquet`` (30-day TTL), so
re-opens are instant and we never hammer the API.

Every network call is best-effort: on any failure the service returns an empty
enrichment dict and the UI falls back to the local catalog data.
"""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import requests

from cinematch.config import SETTINGS
from cinematch.tmdb_net import patch_tmdb_dns

_CACHE_TTL = 30 * 24 * 3600
_TIMEOUT = 15
_CAST_LIMIT = 14
_BASE = "https://api.themoviedb.org/3"
_IMG = "https://image.tmdb.org/t/p"
_CACHE_VERSION = 2


def _empty_payload(source: str) -> dict:
    return {
        "overview": "",
        "tagline": "",
        "runtime_min": None,
        "release_date": "",
        "status": "",
        "backdrop_url": None,
        "genres": [],
        "cast": [],
        "director": [],
        "producers": [],
        "writers": [],
        "seasons": None,
        "episodes": None,
        "trailer_url": None,
        "homepage": None,
        "imdb_id": None,
        "source": source,
    }


class DetailsService:
    """Fetch + cache per-movie TMDB detail enrichments."""

    _COLS = [
        "movie_id", "tmdb_id", "media_type", "fetched_at", "overview", "tagline",
        "runtime_min", "release_date", "status", "backdrop_url", "genres",
        "cast_json", "crew_json", "source", "seasons", "episodes",
        "trailer_url", "homepage", "imdb_id", "version",
    ]

    def __init__(self, movies: pd.DataFrame):
        patch_tmdb_dns()
        self.movies = movies
        self.api_key = SETTINGS.tmdb_api_key
        self._http = requests.Session()
        self._http.trust_env = False
        self._cache_path = SETTINGS.paths.processed_dir / "details_cache.parquet"
        self._cache: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._catalog_overviews: dict[int, str] = {}
        self._load_cache()
        self._load_catalog_overviews()

    # -- loading ------------------------------------------------------------

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            df = pd.read_parquet(self._cache_path)
        except Exception:
            return
        if df.empty:
            return
        for r in df.itertuples(index=False):
            try:
                mid = int(r.movie_id)
            except (TypeError, ValueError):
                continue
            g = getattr(r, "genres", None)
            genres = []
            if g is not None and not (isinstance(g, float) and pd.isna(g)):
                genres = list(g)
            cast_json = getattr(r, "cast_json", None)
            crew_json = getattr(r, "crew_json", None)
            cast = json.loads(cast_json) if isinstance(cast_json, str) and cast_json else []
            crew = json.loads(crew_json) if isinstance(crew_json, str) and crew_json else []
            trailer = getattr(r, "trailer_url", "") or ""
            homepage = getattr(r, "homepage", "") or ""
            imdb_id = getattr(r, "imdb_id", "") or ""
            self._cache[mid] = {
                "tmdb_id": None if pd.isna(r.tmdb_id) else int(r.tmdb_id),
                "media_type": str(r.media_type or "movie"),
                "fetched_at": int(r.fetched_at or 0),
                "overview": str(r.overview or ""),
                "tagline": str(r.tagline or ""),
                "runtime_min": None if pd.isna(r.runtime_min) else float(r.runtime_min),
                "release_date": str(r.release_date or ""),
                "status": str(r.status or ""),
                "backdrop_url": str(r.backdrop_url or "") or None,
                "genres": genres,
                "cast": cast,
                "crew": crew,
                "source": str(r.source or "tmdb"),
                "seasons": None if pd.isna(r.seasons) else int(r.seasons),
                "episodes": None if pd.isna(r.episodes) else int(r.episodes),
                "trailer_url": str(trailer) or None,
                "homepage": str(homepage) or None,
                "imdb_id": str(imdb_id) or None,
                "version": int(getattr(r, "version", 1) or 1),
            }

    def _save_cache(self) -> None:
        if not self._cache:
            return
        rows = []
        for mid, c in self._cache.items():
            rows.append(
                {
                    "movie_id": mid,
                    "tmdb_id": c.get("tmdb_id"),
                    "media_type": c.get("media_type", "movie"),
                    "fetched_at": c.get("fetched_at", int(time.time())),
                    "overview": c.get("overview", ""),
                    "tagline": c.get("tagline", ""),
                    "runtime_min": c.get("runtime_min"),
                    "release_date": c.get("release_date", ""),
                    "status": c.get("status", ""),
                    "backdrop_url": c.get("backdrop_url"),
                    "genres": list(c.get("genres") or []),
                    "cast_json": json.dumps(c.get("cast") or [], ensure_ascii=False),
                    "crew_json": json.dumps(c.get("crew") or [], ensure_ascii=False),
                    "source": c.get("source", "tmdb"),
                    "seasons": c.get("seasons"),
                    "episodes": c.get("episodes"),
                    "trailer_url": c.get("trailer_url"),
                    "homepage": c.get("homepage"),
                    "imdb_id": c.get("imdb_id"),
                    "version": _CACHE_VERSION,
                }
            )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=self._COLS).to_parquet(self._cache_path)

    def _load_catalog_overviews(self) -> None:
        path = SETTINGS.paths.processed_dir / "tmdb_catalog.parquet"
        if not path.exists():
            return
        try:
            cat = pd.read_parquet(path)
        except Exception:
            return
        for r in cat.itertuples(index=False):
            text = getattr(r, "overview", "")
            if text and not (isinstance(text, float) and pd.isna(text)):
                self._catalog_overviews[int(r.tmdb_id)] = str(text).strip()

    # -- internals ------------------------------------------------------------

    def _row(self, movie_id: int) -> pd.Series | None:
        sub = self.movies[self.movies["movie_id"] == movie_id]
        return sub.iloc[0] if not sub.empty else None

    def _search(self, title: str, year: int | None, media_type: str) -> int | None:
        if media_type == "series":
            path, key = "/search/tv", "first_air_date_year"
        else:
            path, key = "/search/movie", "year"
        params = {"api_key": self.api_key, "query": title, "include_adult": "false"}
        if year:
            params[key] = year
        try:
            resp = self._http.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if not results:
                return None
            return int(results[0]["id"])
        except (requests.RequestException, ValueError, KeyError):
            return None

    def _fetch(self, tmdb_id: int, media_type: str) -> dict:
        kind = "tv" if media_type == "series" else "movie"
        try:
            resp = self._http.get(
                f"{_BASE}/{kind}/{tmdb_id}",
                params={"api_key": self.api_key, "language": "en-US", "append_to_response": "credits,videos"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return {}

        cast = []
        for p in (data.get("credits") or {}).get("cast", [])[: _CAST_LIMIT + 4]:
            cast.append(
                {
                    "id": int(p["id"]) if p.get("id") else None,
                    "name": str(p.get("name") or ""),
                    "character": str(p.get("character") or ""),
                    "profile_url": (
                        f"{_IMG}/w185{p['profile_path']}" if p.get("profile_path") else None
                    ),
                }
            )
        crew = [
            {
                "id": int(c.get("id")) if c.get("id") else None,
                "name": str(c.get("name") or ""),
                "job": str(c.get("job") or ""),
                "department": str(c.get("department") or ""),
                "profile_url": (
                    f"{_IMG}/w185{c['profile_path']}" if c.get("profile_path") else None
                ),
            }
            for c in (data.get("credits") or {}).get("crew", [])
        ]
        trailer_url = None
        for v in (data.get("videos") or {}).get("results") or []:
            if v.get("site") == "YouTube" and v.get("key"):
                trailer_url = f"https://www.youtube.com/watch?v={v['key']}"
                break
        return {
            "overview": str(data.get("overview") or "").strip(),
            "tagline": str(data.get("tagline") or "").strip(),
            "runtime_min": data.get("runtime"),
            "release_date": str(data.get("release_date") or data.get("first_air_date") or ""),
            "status": str(data.get("status") or ""),
            "backdrop_url": f"{_IMG}/w780{data['backdrop_path']}" if data.get("backdrop_path") else None,
            "genres": [g["name"] for g in (data.get("genres") or []) if g.get("name")],
            "cast": cast,
            "crew": crew,
            "seasons": len(data.get("seasons") or []) or None,
            "episodes": data.get("number_of_episodes"),
            "trailer_url": trailer_url,
            "homepage": str(data.get("homepage") or "") or None,
            "imdb_id": str(data.get("imdb_id") or "") or None,
        }

    def _to_payload(self, c: dict) -> dict:
        crew = c.get("crew") or []
        cast = c.get("cast") or []
        return {
            "tmdb_id": c.get("tmdb_id"),
            "overview": c.get("overview", ""),
            "tagline": c.get("tagline", ""),
            "runtime_min": c.get("runtime_min"),
            "release_date": c.get("release_date", ""),
            "status": c.get("status", ""),
            "backdrop_url": c.get("backdrop_url"),
            "genres": list(c.get("genres") or []),
            "cast": cast,
            "director": sorted({x["name"] for x in crew if x.get("job") == "Director"}),
            "producers": sorted(
                {
                    x["name"]
                    for x in crew
                    if x.get("department") == "Production" and "Producer" in x.get("job", "")
                }
            ),
            "writers": sorted(
                {
                    x["name"]
                    for x in crew
                    if x.get("department") == "Writing"
                    and x.get("job") in {"Writer", "Screenplay", "Screenwriter", "Teleplay", "Story"}
                }
            ),
            "seasons": c.get("seasons"),
            "episodes": c.get("episodes"),
            "trailer_url": c.get("trailer_url"),
            "homepage": c.get("homepage"),
            "imdb_id": c.get("imdb_id"),
            "source": c.get("source", "tmdb"),
        }

    # -- public ----------------------------------------------------------------

    def _fresh(self, cached: dict | None) -> bool:
        if not cached:
            return False
        if cached.get("version", 1) < _CACHE_VERSION:
            return False
        return time.time() - cached["fetched_at"] < _CACHE_TTL

    def enrich(self, movie_id: int) -> dict:
        """Return the TMDB enrichment for a movie id (cached, best-effort)."""
        row = self._row(movie_id)
        media_type = str(getattr(row, "media_type", "") or "movie") if row is not None else "movie"
        cached = self._cache.get(movie_id)
        if self._fresh(cached):
            return self._to_payload(cached)

        with self._lock:
            cached = self._cache.get(movie_id)
            if self._fresh(cached):
                return self._to_payload(cached)

            tmdb_id = None
            if row is not None:
                t = getattr(row, "tmdb_id", None)
                if t is not None and not (isinstance(t, float) and pd.isna(t)):
                    tmdb_id = int(t)
            if tmdb_id is None:
                title = getattr(row, "clean_title", "") if row is not None else ""
                year = getattr(row, "year", None)
                if title and (not (isinstance(year, float) and pd.isna(year))):
                    tmdb_id = self._search(str(title), int(year) if year else None, media_type)
            if tmdb_id is None:
                entry = _empty_payload("none")
                entry.update(
                    {
                        "movie_id": movie_id,
                        "tmdb_id": None,
                        "media_type": media_type,
                        "fetched_at": int(time.time()),
                        "cast": [],
                        "crew": [],
                        "version": _CACHE_VERSION,
                    }
                )
                self._cache[movie_id] = entry
                self._save_cache()
                return self._to_payload(entry)

            fetched = self._fetch(tmdb_id, media_type)
            overview = fetched.get("overview") or self._catalog_overviews.get(tmdb_id, "")
            fetched["overview"] = overview
            entry = {
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "media_type": media_type,
                "fetched_at": int(time.time()),
                **fetched,
                "source": "tmdb",
                "version": _CACHE_VERSION,
            }
            self._cache[movie_id] = entry
            self._save_cache()
            return self._to_payload(entry)

    def clear(self) -> None:
        self._cache.clear()
        self._save_cache()
