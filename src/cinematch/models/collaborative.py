"""Collaborative filtering via truncated SVD (matrix factorization).

Implements the classic bias-aware SVD baseline:

    r_hat(u, i) = global_mean + user_bias[u] + item_bias[i] + u_u . v_i

where ``u_u . v_i`` is the dot product of latent vectors recovered from a
``TruncatedSVD`` on the mean-centered rating matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class SVDFactorizer:
    """Matrix-factorization recommender trained on a ratings frame."""

    def __init__(self, n_components: int = 50, seed: int = 42):
        self.n_components = n_components
        self.seed = seed

    def fit(self, ratings: pd.DataFrame) -> "SVDFactorizer":
        ratings = ratings[["user_id", "movie_id", "rating"]].dropna()
        self.user_ids = sorted(ratings["user_id"].unique().tolist())
        self.movie_ids = sorted(ratings["movie_id"].unique().tolist())
        self.u_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.i_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        rows = ratings["user_id"].map(self.u_index).to_numpy()
        cols = ratings["movie_id"].map(self.i_index).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=float)

        self.R = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids))
        )
        self.global_mean = float(self.R.data.mean())

        user_counts = np.asarray(self.R.getnnz(axis=1)).ravel()
        user_sums = np.asarray(self.R.sum(axis=1)).ravel()
        user_means = user_sums / np.maximum(user_counts, 1)
        self.user_bias = user_means - self.global_mean

        item_counts = np.asarray(self.R.getnnz(axis=0)).ravel()
        item_sums = np.asarray(self.R.sum(axis=0)).ravel()
        item_means = item_sums / np.maximum(item_counts, 1)
        self.item_bias = item_means - self.global_mean

        residual = vals - (
            self.global_mean + self.user_bias[rows] + self.item_bias[cols]
        )
        R_residual = sparse.csr_matrix(
            (residual, (rows, cols)), shape=self.R.shape
        )

        svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
        self.U = svd.fit_transform(R_residual)  # (n_users, k)
        self.Vt = svd.components_  # (k, n_items)
        self.explained_variance = float(svd.explained_variance_ratio_.sum())
        return self

    def predict_all_items(self, user_index: int) -> np.ndarray:
        """Predicted scores for every item for a trained user row."""
        latent = self.U[user_index] @ self.Vt
        return self.global_mean + self.user_bias[user_index] + self.item_bias + latent

    def top_for_user(
        self,
        user_id: int,
        k: int = 20,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Ranked (movie_id, score) list for a user, skipping excluded ids."""
        if user_id not in self.u_index:
            return []
        exclude = exclude or set()
        scores = self.predict_all_items(self.u_index[user_id])
        order = np.argsort(-scores)
        out: list[tuple[int, float]] = []
        for idx in order:
            movie_id = self.movie_ids[idx]
            if movie_id in exclude:
                continue
            out.append((movie_id, float(scores[idx])))
            if len(out) >= k:
                break
        return out

    def save(self, path) -> None:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "SVDFactorizer":
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)


def popularity_scores(
    ratings: pd.DataFrame, prior_rating: float = 3.7, prior_count: int = 10
) -> dict[int, float]:
    """Bayesian-average popularity for every movie.

    ``score = (sum + prior_rating*prior_count) / (count + prior_count)``
    which smooths low-count movies toward the global prior.
    """
    grouped = ratings.groupby("movie_id")["rating"]
    sums = grouped.sum()
    counts = grouped.count()
    scores = (sums + prior_rating * prior_count) / (counts + prior_count)
    return scores.to_dict()


class ItemBasedCF:
    """Item-based collaborative filtering via co-rating cosine similarity.

    For a user the predicted score of an unseen movie ``j`` is the sum of
    similarity between ``j`` and the movies the user rated, weighted by the
    user's mean-centered ratings:

        score(u, j) = sum_i sim(j, i) * (r_ui - user_mean_u)

    which captures "recommend things similar to what you already liked".
    """

    def __init__(self, top_sim: int = 100):
        self.top_sim = top_sim

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        ratings = ratings[["user_id", "movie_id", "rating"]].dropna()
        self.user_ids = sorted(ratings["user_id"].unique().tolist())
        self.movie_ids = sorted(ratings["movie_id"].unique().tolist())
        self.u_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.i_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        rows = ratings["user_id"].map(self.u_index).to_numpy()
        cols = ratings["movie_id"].map(self.i_index).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=float)

        R = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids))
        )

        user_counts = np.asarray(R.getnnz(axis=1)).ravel()
        user_sums = np.asarray(R.sum(axis=1)).ravel()
        self.user_means = user_sums / np.maximum(user_counts, 1)

        # Mean-center each user's ratings to remove individual bias.
        centered = R.copy().astype(float)
        centered.data -= np.repeat(self.user_means, np.diff(centered.indptr))
        self._centered = centered

        # Item-item cosine similarity = (L2-normalized item columns) dot product.
        col_norm = np.sqrt(
            np.asarray(centered.multiply(centered).sum(axis=0)).ravel()
        )
        col_norm[col_norm == 0] = 1.0
        normalized = centered @ sparse.diags(1.0 / col_norm)
        self.sim = (normalized.T @ normalized).toarray()
        np.fill_diagonal(self.sim, 0.0)

        # Keep only the top-K similar neighbours per item (sparsity + speed).
        if self.top_sim and self.top_sim < self.sim.shape[0]:
            k = self.top_sim
            top_idx = np.argsort(-self.sim, axis=1)[:, :k]
            mask = np.zeros_like(self.sim)
            np.put_along_axis(mask, top_idx, 1.0, axis=1)
            self.sim *= mask
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        """Predicted score for every item (user-centered ratings @ similarity)."""
        row = self._centered.getrow(self.u_index[user_id])
        return np.asarray(row.toarray()).ravel() @ self.sim

    def top_for_user(
        self,
        user_id: int,
        k: int = 20,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Ranked (movie_id, score) list for a user, skipping excluded ids."""
        if user_id not in self.u_index:
            return []
        exclude = exclude or set()
        scores = self.score_all(user_id)
        order = np.argsort(-scores)
        out: list[tuple[int, float]] = []
        for idx in order:
            movie_id = self.movie_ids[idx]
            if movie_id in exclude:
                continue
            out.append((movie_id, float(scores[idx])))
            if len(out) >= k:
                break
        return out

    def score_items(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        """Scores for an explicit subset of items (used by offline evaluation)."""
        if user_id not in self.u_index:
            return np.zeros(len(item_ids), dtype=float)
        scores = self.score_all(user_id)
        return np.array([scores[self.i_index[i]] for i in item_ids], dtype=float)

    def save(self, path) -> None:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "ItemBasedCF":
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)
