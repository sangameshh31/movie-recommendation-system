"""Real-time feedback loop.

A per-user store of signals (like / dislike / watchlist / watched / star rating)
persisted to ``data/feedback.json`` so the account library survives restarts.
The hybrid ranker uses these signals to dynamically re-rank results on every
request.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path

import pandas as pd

from cinematch.config import SETTINGS

_BUCKETS = ("liked", "disliked", "watchlist", "watched")


class FeedbackStore:
    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else SETTINGS.paths.data_dir / "feedback.json"
        self._lock = threading.RLock()
        self._liked: dict[int, set[int]] = defaultdict(set)
        self._disliked: dict[int, set[int]] = defaultdict(set)
        self._watchlist: dict[int, set[int]] = defaultdict(set)
        self._watched: dict[int, set[int]] = defaultdict(set)
        self._stars: dict[int, dict[int, float]] = defaultdict(dict)
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        for bucket in _BUCKETS:
            for user, ids in (raw.get(bucket) or {}).items():
                target = getattr(self, f"_{bucket}")
                target[int(user)] = {int(mid) for mid in ids}
        for user, stars in (raw.get("stars") or {}).items():
            self._stars[int(user)] = {int(mid): float(v) for mid, v in stars.items()}

    def _save(self) -> None:
        payload = {
            bucket: {str(u): sorted(ids) for u, ids in getattr(self, f"_{bucket}").items()}
            for bucket in _BUCKETS
        }
        payload["stars"] = {
            str(u): {str(mid): v for mid, v in stars.items()}
            for u, stars in self._stars.items()
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # -- mutations -------------------------------------------------------
    def record(self, user_id: int, movie_id: int, action: str, value: float | None = None) -> None:
        user_id, movie_id = int(user_id), int(movie_id)
        with self._lock:
            if action == "rate":
                self._stars[user_id][movie_id] = float(value) if value is not None else 5.0
            elif action in ("unrate", "remove"):
                self._stars[user_id].pop(movie_id, None)
                for bucket in _BUCKETS:
                    getattr(self, f"_{bucket}")[user_id].discard(movie_id)
            elif action in ("like", "dislike", "watchlist", "watched", "unwatched"):
                for bucket in ("liked", "disliked", "watchlist", "watched"):
                    getattr(self, f"_{bucket}")[user_id].discard(movie_id)
                bucket = {
                    "like": "liked",
                    "dislike": "disliked",
                    "watchlist": "watchlist",
                    "watched": "watched",
                }.get(action)
                if bucket:
                    getattr(self, f"_{bucket}")[user_id].add(movie_id)
                if action == "watched":
                    self._stars[user_id].pop(movie_id, None)
            else:
                raise ValueError(f"Unknown feedback action: {action!r}")
            self._save()

    def _discard_from_all(self, user_id: int, movie_id: int) -> None:
        with self._lock:
            for bucket in _BUCKETS:
                getattr(self, f"_{bucket}")[user_id].discard(movie_id)
            self._stars[user_id].pop(movie_id, None)

    def unmark(self, user_id: int, movie_id: int) -> None:
        """Clear every signal for a movie (treated like 'remove')."""
        self._discard_from_all(user_id, movie_id)
        self._save()

    # -- reads -----------------------------------------------------------
    def liked(self, user_id: int) -> set[int]:
        return set(self._liked.get(user_id, ()))

    def disliked(self, user_id: int) -> set[int]:
        return set(self._disliked.get(user_id, ()))

    def watchlist(self, user_id: int) -> set[int]:
        return set(self._watchlist.get(user_id, ()))

    def watched(self, user_id: int) -> set[int]:
        return set(self._watched.get(user_id, ()))

    def stars(self, user_id: int) -> dict[int, float]:
        return dict(self._stars.get(user_id, {}))

    def profile(self, user_id: int) -> dict:
        """All signals for one user (used by the library endpoint)."""
        return {
            "liked": sorted(self.liked(user_id)),
            "disliked": sorted(self.disliked(user_id)),
            "watchlist": sorted(self.watchlist(user_id)),
            "watched": sorted(self.watched(user_id)),
            "stars": self.stars(user_id),
        }

    def genre_overlap_deltas(
        self, user_id: int, movies: pd.DataFrame, boost: float | None = None
    ) -> dict[int, float]:
        """Per-movie score deltas derived from liked/disliked genres.

        A movie is boosted when it shares genres with liked films and punished
        when it overlaps the disliked set. Watchlisted items get a small bump.
        """
        boost = SETTINGS.retrieval.feedback_boost if boost is None else boost
        liked = self.liked(user_id)
        disliked = self.disliked(user_id)
        watch = self.watchlist(user_id)

        if not (liked or disliked or watch):
            return {}

        genre_of = {
            int(row.movie_id): set(row.genres) for row in movies.itertuples(index=False)
        }
        liked_genres: set[str] = set()
        for mid in liked:
            liked_genres |= genre_of.get(mid, set())
        disliked_genres: set[str] = set()
        for mid in disliked:
            disliked_genres |= genre_of.get(mid, set())

        deltas: dict[int, float] = {}
        for mid, genres in genre_of.items():
            delta = 0.0
            if liked_genres:
                delta += boost * len(genres & liked_genres) / max(len(liked_genres), 1)
            if disliked_genres:
                delta -= boost * len(genres & disliked_genres) / max(len(disliked_genres), 1)
            if mid in watch:
                delta += 0.5 * boost
            if delta:
                deltas[mid] = delta
        return deltas
