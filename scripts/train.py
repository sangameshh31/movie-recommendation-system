"""Train (or reload) the collaborative-filtering model and report stats.

    python scripts/train.py [--force]
"""

import argparse
import time

from cinematch.config import SETTINGS
from cinematch.data import load_processed
from cinematch.models import SVDFactorizer


def main():
    parser = argparse.ArgumentParser(description="Train the SVD collaborative model.")
    parser.add_argument("--force", action="store_true", help="Retrain even if cached")
    args = parser.parse_args()

    path = SETTINGS.paths.models_dir / f"svd_{SETTINGS.data.size}.pkl"
    if path.exists() and not args.force:
        print(f"Cached model found at {path} — use --force to retrain.")
        return

    data = load_processed(paths=SETTINGS.paths)
    ratings = data["ratings"]
    print(f"Training SVD on {len(ratings):,} ratings ...")

    t0 = time.time()
    svd = SVDFactorizer(
        n_components=SETTINGS.retrieval.svd_factors, seed=SETTINGS.data.random_seed
    ).fit(ratings)
    svd.save(path)
    print(
        f"Trained in {time.time() - t0:.1f}s: {svd.U.shape[0]:,} users x "
        f"{svd.Vt.shape[1]:,} movies | explained variance {svd.explained_variance:.2%}"
    )


if __name__ == "__main__":
    main()
