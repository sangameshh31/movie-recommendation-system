"""Fast unit tests for the pure-logic components (no network / no data download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cinematch.feedback import FeedbackStore
from cinematch.models.collaborative import SVDFactorizer, ItemBasedCF, popularity_scores
from cinematch.models.hybrid import hybrid_rerank, minmax_normalize
from cinematch.text import build_movie_text, build_query_text


def _fake_ratings() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    users = [1, 2, 3, 4, 5]
    movies = list(range(1, 11))
    rows = []
    for u in users:
        for m in movies:
            rows.append((u, m, float(rng.integers(1, 6))))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"])


def test_minmax_normalize():
    values = {"a": 1.0, "b": 3.0, "c": 5.0}
    norm = minmax_normalize(values)
    assert norm["a"] == pytest.approx(0.0)
    assert norm["b"] == pytest.approx(0.5)
    assert norm["c"] == pytest.approx(1.0)
    assert minmax_normalize({"x": 2.0})["x"] == pytest.approx(0.5)


def test_hybrid_rerank_orders_by_weights():
    candidates = {
        1: {"cf": 5.0, "cb": 0.1},
        2: {"cf": 4.0, "cb": 0.9},
    }
    ranked = hybrid_rerank(candidates, weights={"cf": 1.0, "cb": 0.0})
    assert ranked[0][0] == 1
    ranked = hybrid_rerank(candidates, weights={"cf": 0.0, "cb": 1.0})
    assert ranked[0][0] == 2


def test_svd_factorizer_recovers_signal():
    ratings = _fake_ratings()
    svd = SVDFactorizer(n_components=4, seed=42).fit(ratings)
    recs = svd.top_for_user(user_id=1, k=5, exclude={1, 2, 3})
    assert len(recs) == 5
    assert all(0.0 < score for _, score in recs)
    # Cold-start user gets nothing from CF path.
    assert svd.top_for_user(user_id=999, k=5) == []


def test_itembased_cf_ranks_similar_items_first():
    ratings = pd.DataFrame(
        [
            # Three users share the same taste cluster.
            (1, 1, 5.0), (1, 2, 5.0), (1, 3, 4.0),
            (2, 1, 5.0), (2, 2, 4.0), (2, 3, 4.0),
            (3, 1, 4.0), (3, 2, 5.0), (3, 3, 5.0),
        ],
        columns=["user_id", "movie_id", "rating"],
    )
    model = ItemBasedCF(top_sim=5).fit(ratings)
    recs = dict(model.top_for_user(user_id=1, k=10, exclude={1, 2}))
    assert set(recs) == {3}
    # Cold-start user gets nothing from the item-CF path.
    assert model.top_for_user(user_id=999, k=5) == []


def test_popularity_scores_are_bounded():
    ratings = pd.DataFrame(
        [
            (1, 1, 5.0), (2, 1, 5.0),
            (1, 2, 1.0), (2, 2, 1.0),
            (1, 3, 5.0), (2, 3, 5.0), (3, 3, 1.0),
        ],
        columns=["user_id", "movie_id", "rating"],
    )
    scores = popularity_scores(ratings)
    assert all(1.0 <= s <= 5.0 for s in scores.values())
    assert scores[1] > scores[2]  # consistently-high movie ranks above consistently-low

    # Identical rating distributions must yield identical scores.
    equal = pd.DataFrame(
        [(1, 1, 4.0), (2, 1, 2.0), (1, 2, 4.0), (2, 2, 2.0)],
        columns=["user_id", "movie_id", "rating"],
    )
    eq = popularity_scores(equal)
    assert eq[1] == eq[2]


def test_feedback_deltas_boost_similar_genres():
    store = FeedbackStore()
    movies = pd.DataFrame(
        [
            {"movie_id": 1, "genres": ["Action", "Thriller"]},
            {"movie_id": 2, "genres": ["Action"]},
            {"movie_id": 3, "genres": ["Romance"]},
        ]
    )
    store.record(7, 1, "like")
    deltas = store.genre_overlap_deltas(7, movies)
    assert deltas[2] > 0  # shares Action
    assert deltas.get(3, 0.0) <= 0.0  # no overlap -> not boosted


def test_feedback_rejects_unknown_action():
    store = FeedbackStore()
    with pytest.raises(ValueError):
        store.record(1, 2, "nope")


def test_build_movie_text():
    movies = pd.DataFrame(
        [
            {"movie_id": 1, "title": "Interstellar (2014)", "clean_title": "Interstellar",
             "year": 2014, "genres": ["Sci-Fi", "Adventure"]},
        ]
    )
    enriched = build_movie_text(movies)
    assert "Interstellar (2014)" in enriched.iloc[0]["text"]
    assert "Sci-Fi" in enriched.iloc[0]["text"]
    assert build_query_text("deep sea", ["Drama"]) == "deep sea :: Drama"
