"""Director / cast exploration via TMDB.

Given a person's name, resolve them on TMDB (search person + profile + combined
credits), then map any of their credits that exist in our catalog back to the
local movie payloads so the UI can offer "more from this person" rails.

Results are cached to ``data/processed/people_cache.parquet`` (7-day TTL).
Every network call is best-effort: failures return ``None`` and the UI hides
the feature gracefully.
"""

from __future__ import annotations

import json
import re
import threading
import time

import pandas as pd
import requests

from cinematch.config import SETTINGS
from cinematch.tmdb_net import patch_tmdb_dns

_CACHE_TTL = 7 * 24 * 3600
_TIMEOUT = 15
_BASE = "https://api.themoviedb.org/3"
_IMG = "https://image.tmdb.org/t/p"


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


class PeopleService:
    _COLS = ["key", "fetched_at", "payload_json"]

    def __init__(self, movies: pd.DataFrame):
        patch_tmdb_dns()
        self.movies = movies
        self.api_key = SETTINGS.tmdb_api_key
        self._http = requests.Session()
        self._http.trust_env = False
        self._cache_path = SETTINGS.paths.processed_dir / "people_cache.parquet"
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._tmdb_to_movie: dict[int, int] = {}
        self._load_cache()
        self._build_tmdb_map()

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
                self._cache[str(r.key)] = {
                    "fetched_at": int(r.fetched_at or 0),
                    "payload": json.loads(r.payload_json),
                }
            except (ValueError, TypeError):
                continue

    def _save_cache(self) -> None:
        if not self._cache:
            return
        rows = [
            {
                "key": key,
                "fetched_at": entry["fetched_at"],
                "payload_json": json.dumps(entry["payload"], ensure_ascii=False),
            }
            for key, entry in self._cache.items()
        ]
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=self._COLS).to_parquet(self._cache_path)

    def _build_tmdb_map(self) -> None:
        if "tmdb_id" not in self.movies.columns:
            return
        sub = self.movies[["movie_id", "tmdb_id"]].dropna(subset=["tmdb_id"])
        self._tmdb_to_movie = {
            int(r.tmdb_id): int(r.movie_id) for r in sub.itertuples(index=False)
        }

    # -- TMDB ----------------------------------------------------------------

    def _search_person(self, name: str) -> dict | None:
        try:
            resp = self._http.get(
                f"{_BASE}/search/person",
                params={"api_key": self.api_key, "query": name, "include_adult": "false"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except (requests.RequestException, ValueError):
            return None
        if not results:
            return None
        return results[0]

    def _person(self, person_id: int) -> dict:
        try:
            resp = self._http.get(
                f"{_BASE}/person/{person_id}",
                params={"api_key": self.api_key, "language": "en-US"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            return {}

    def _combined_credits(self, person_id: int) -> dict:
        try:
            resp = self._http.get(
                f"{_BASE}/person/{person_id}/combined_credits",
                params={"api_key": self.api_key, "language": "en-US"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            return {}

    @staticmethod
    def _year(credit: dict) -> int | None:
        raw = credit.get("release_date") or credit.get("first_air_date") or ""
        m = re.search(r"(\d{4})", str(raw))
        return int(m.group(1)) if m else None

    @staticmethod
    def _poster(credit: dict) -> str | None:
        path = credit.get("poster_path")
        return f"{_IMG}/w185{path}" if path else None

    # -- public ----------------------------------------------------------------

    def search(self, name: str, get_movie) -> dict | None:
        """Resolve a person and their catalog-relevant credits.

        ``get_movie`` is the service's ``get_movie(movie_id)`` callback, kept as
        a parameter to avoid a circular import.
        """
        name = (name or "").strip()
        if not name:
            return None
        key = _normalize(name)
        cached = self._cache.get(key)
        if cached and time.time() - cached["fetched_at"] < _CACHE_TTL:
            return cached["payload"]

        with self._lock:
            cached = self._cache.get(key)
            if cached and time.time() - cached["fetched_at"] < _CACHE_TTL:
                return cached["payload"]

            result = self._search_person(name)
            if not result:
                return None
            person_id = int(result["id"])
            profile = self._person(person_id)

            known_for: list[dict] = []
            catalog_movies: list[dict] = []
            seen: set[int] = set()

            credits = self._combined_credits(person_id)
            for credit in (credits.get("cast") or []) + (credits.get("crew") or []):
                tmdb_id = int(credit.get("id")) if credit.get("id") else None
                if not tmdb_id:
                    continue
                item = {
                    "tmdb_id": tmdb_id,
                    "title": str(credit.get("title") or credit.get("name") or ""),
                    "year": self._year(credit),
                    "media_type": "series" if credit.get("media_type") == "tv" else "movie",
                    "poster_url": self._poster(credit),
                    "vote_average": (
                        round(float(credit["vote_average"]), 1)
                        if credit.get("vote_average")
                        else None
                    ),
                    "role": str(credit.get("character") or credit.get("job") or ""),
                }
                if item["title"]:
                    known_for.append(item)
                local = self._tmdb_to_movie.get(tmdb_id)
                if local and local not in seen:
                    seen.add(local)
                    movie = get_movie(local)
                    if movie:
                        movie["tmdb_id"] = tmdb_id
                        catalog_movies.append(movie)

            known_for.sort(key=lambda c: (c["vote_average"] or 0.0), reverse=True)

            payload = {
                "name": str(result.get("name") or name),
                "profile_url": (
                    f"{_IMG}/w500{result['profile_path']}" if result.get("profile_path") else None
                ),
                "department": str(profile.get("known_for_department") or ""),
                "biography": str(profile.get("biography") or "").strip(),
                "catalog_movies": catalog_movies,
                "known_works": known_for[:12],
            }
            self._cache[key] = {"fetched_at": int(time.time()), "payload": payload}
            self._save_cache()
            return payload
