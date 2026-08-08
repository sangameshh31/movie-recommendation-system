"""Build the lightweight runtime assets for 1 GB Streamlit Cloud deploys.

The cloud container skips Qdrant + torch/sentence-transformers entirely, so it
needs precomputed assets:

* ``data/processed/light_index.pkl`` — the same semantic embeddings that feed
  the Qdrant index (so runtime query vectors are comparable) plus their movie
  payloads, stored as plain numpy for cosine search at runtime.
* ``models_cache/svd_<size>.pkl`` — collaborative SVD on the full catalog
  (trained by this script if not already present).

Run (with the full dev environment):  python scripts/build_light_assets.py
"""

import pickle
import time

import numpy as np
import pandas as pd

from cinematch.config import SETTINGS
from cinematch.data import load_processed
from cinematch.embeddings import embed_texts
from cinematch.models import SVDFactorizer
from cinematch.text import build_movie_text


def main() -> None:
    t0 = time.time()
    data = load_processed(paths=SETTINGS.paths)
    movies, ratings = data["movies"], data["ratings"]
    print(f"Loaded {len(movies):,} movies / {len(ratings):,} ratings")

    # --- semantic index (same text + model as scripts/index_vectors.py) -----
    enriched = build_movie_text(movies, None)
    vectors = embed_texts(enriched["text"].tolist())
    ids = enriched["movie_id"].astype(np.int64).to_numpy()

    payloads = []
    for row in enriched.itertuples(index=False):
        poster = getattr(row, "poster_url", None)
        poster_url = (
            ""
            if poster is None
            or (isinstance(poster, float) and pd.isna(poster))
            or not str(poster).strip()
            else str(poster)
        )
        vote = getattr(row, "vote_average", None)
        payloads.append(
            {
                "movie_id": int(row.movie_id),
                "title": row.title,
                "clean_title": row.clean_title,
                "year": None if pd.isna(row.year) else int(row.year),
                "genres": list(row.genres),
                "language": str(getattr(row, "language", "") or ""),
                "origin": str(getattr(row, "origin", "") or ""),
                "media_type": str(getattr(row, "media_type", "") or "movie"),
                "vote_average": (
                    None if vote is None or pd.isna(vote) else round(float(vote), 1)
                ),
                "poster_url": poster_url,
            }
        )

    index_path = SETTINGS.paths.processed_dir / "light_index.pkl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as fh:
        pickle.dump(
            {"ids": ids, "vectors": vectors, "payloads": payloads},
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(
        f"Saved {index_path} ({vectors.shape[0]} x {vectors.shape[1]}) in "
        f"{time.time() - t0:.1f}s"
    )

    # --- collaborative SVD ---------------------------------------------------
    svd_path = SETTINGS.paths.models_dir / f"svd_{SETTINGS.data.size}.pkl"
    if svd_path.exists():
        print(f"SVD already present at {svd_path} (skip)")
    else:
        svd = SVDFactorizer(
            n_components=SETTINGS.retrieval.svd_factors,
            seed=SETTINGS.data.random_seed,
        ).fit(ratings)
        svd.save(svd_path)
        print(
            f"Trained SVD: {svd.U.shape[0]:,} users x {svd.Vt.shape[1]:,} movies "
            f"(explained variance {svd.explained_variance:.2%}) -> {svd_path}"
        )


if __name__ == "__main__":
    main()
