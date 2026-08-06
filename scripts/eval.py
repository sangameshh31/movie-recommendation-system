"""Offline evaluation: Hit Rate@k and Mean Reciprocal Rank (MRR).

Protocol (NCF-style negative sampling, the de-facto standard for implicit
recommendation evaluation):

1. Leave one rating out per user (uniform random hold-out).
2. For each held-out (user, item) pair sample 99 negative items the user has
   never rated.
3. Rank the 100 candidate items (1 positive + 99 negatives) with the hybrid
   collaborative signal (SVD + item-based CF + popularity).
4. A hit occurs when the positive item is ranked in the top-k.

With 99 negatives the random-chance Hit Rate@10 is exactly 0.10.

    python scripts/eval.py [--topk 10] [--n-negatives 99]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cinematch.config import SETTINGS
from cinematch.data import load_processed
from cinematch.models import SVDFactorizer, ItemBasedCF
from cinematch.models.collaborative import popularity_scores
from cinematch.models.hybrid import minmax_normalize


def leave_one_out_split(ratings: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one uniformly-random rating per user for testing."""
    rng = np.random.default_rng(seed)
    test_rows, train_rows = [], []
    for _, group in ratings.groupby("user_id"):
        keep = int(rng.integers(0, len(group)))
        test_rows.append(group.iloc[[keep]])
        train_rows.append(group.drop(group.index[keep]))
    return pd.concat(train_rows), pd.concat(test_rows)


def rank_hybrid(
    svd: SVDFactorizer,
    itemcf: ItemBasedCF,
    popularity: dict[int, float],
    user_id: int,
    items: list[int],
    w_svd: float = 0.1,
    w_icf: float = 0.7,
    w_pop: float = 0.2,
) -> np.ndarray:
    """Min-max normalized blend of SVD + itemCF + popularity over ``items``."""
    icf_scores = itemcf.score_items(user_id, items)

    if user_id in svd.u_index:
        all_svd = svd.predict_all_items(svd.u_index[user_id])
        svd_scores = np.array([all_svd[svd.i_index[i]] for i in items])
    else:
        svd_scores = np.zeros(len(items))

    pop_scores = np.array([popularity.get(i, 0.0) for i in items])

    def _norm(values: np.ndarray) -> np.ndarray:
        normalized = minmax_normalize(dict(enumerate(values.tolist())))
        return np.array([normalized[i] for i in range(len(values))])

    return (
        w_svd * _norm(svd_scores)
        + w_icf * _norm(icf_scores)
        + w_pop * _norm(pop_scores)
    )


def evaluate(
    svd: SVDFactorizer,
    itemcf: ItemBasedCF,
    train: pd.DataFrame,
    test: pd.DataFrame,
    popularity: dict[int, float],
    top_k: int,
    n_negatives: int,
    seed: int = 7,
) -> dict:
    """Compute HitRate@k and MRR over held-out ratings (1 pos + n negatives)."""
    rng = np.random.default_rng(seed)
    all_items = np.sort(train["movie_id"].unique())

    hits = 0
    reciprocal_ranks: list[float] = []
    total = 0

    for user_id, held_out in test.groupby("user_id"):
        user_train = train[train["user_id"] == user_id]
        rated = set(user_train["movie_id"])
        target = int(held_out.iloc[0]["movie_id"])

        neg_candidates = all_items[~np.isin(all_items, list(rated) + [target])]
        negatives = rng.choice(neg_candidates, size=n_negatives, replace=False)
        items = np.concatenate(([target], negatives))

        final_scores = rank_hybrid(svd, itemcf, popularity, int(user_id), items.tolist())
        order = np.argsort(-final_scores)
        # Position of the positive item (index 0) in the descending ranking.
        rank_of_positive = int(np.where(order == 0)[0][0]) + 1  # 1-based

        total += 1
        if rank_of_positive <= top_k:
            hits += 1
        reciprocal_ranks.append(1.0 / rank_of_positive)

    return {
        "users_evaluated": total,
        "negatives_per_user": n_negatives,
        "hit_rate": hits / max(total, 1),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n-negatives", type=int, default=99)
    args = parser.parse_args()

    data = load_processed(paths=SETTINGS.paths)
    ratings = data["ratings"]

    train, test = leave_one_out_split(ratings, seed=SETTINGS.data.random_seed)
    print(f"Train ratings: {len(train):,} | Test ratings: {len(test):,}")

    svd = SVDFactorizer(
        n_components=SETTINGS.retrieval.svd_factors, seed=SETTINGS.data.random_seed
    ).fit(train)
    itemcf = ItemBasedCF().fit(train)
    popularity = popularity_scores(train)

    metrics = evaluate(
        svd, itemcf, train, test, popularity,
        top_k=args.topk, n_negatives=args.n_negatives,
    )
    print(
        f"HitRate@{args.topk}: {metrics['hit_rate']:.3f} "
        f"| MRR: {metrics['mrr']:.3f} "
        f"| users: {metrics['users_evaluated']} "
        f"| negatives: {metrics['negatives_per_user']} "
        f"(random-chance HR@{args.topk} = {args.topk / (args.n_negatives + 1):.2f})"
    )


if __name__ == "__main__":
    main()
