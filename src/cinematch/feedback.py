"""Real-time feedback loop.

An in-memory store of per-user signals (like / dislike / watchlist) that the
hybrid ranker uses to dynamically re-rank results on every request.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from cinematch.config import SETTINGS


class FeedbackStore:
    def __init__(self):
        self._liked: dict[int, set[int]] = defaultdict(set)
        self._disliked: dict[int, set[int]] = defaultdict(set)
        self._watchlist: dict[int, set[int]] = defaultdict(set)

    def record(self, user_id: int, movie_id: int, action: str) -> None:
        user_id, movie_id = int(user_id), int(movie_id)
        for bucket in (self._liked, self._disliked, self._watchlist):
            bucket[user_id].discard(movie_id)
        if action == "like":
            self._liked[user_id].add(movie_id)
        elif action == "dislike":
            self._disliked[user_id].add(movie_id)
        elif action == "watchlist":
            self._watchlist[user_id].add(movie_id)
        elif action != "remove":
            raise ValueError(f"Unknown feedback action: {action!r}")

    def liked(self, user_id: int) -> set[int]:
        return self._liked.get(user_id, set())

    def disliked(self, user_id: int) -> set[int]:
        return self._disliked.get(user_id, set())

    def watchlist(self, user_id: int) -> set[int]:
        return self._watchlist.get(user_id, set())

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
