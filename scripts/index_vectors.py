"""Build semantic embeddings and index them into the vector store.

Run after ``download_data.py`` (or the first time you run the API).

    python scripts/index_vectors.py [--reset]
"""

import argparse
import time

from cinematch.config import SETTINGS
from cinematch.data import load_processed
from cinematch.embeddings import embed_texts
from cinematch.text import build_movie_text, make_tmdb_fetcher
from cinematch.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Embed movies and index into Qdrant.")
    parser.add_argument("--reset", action="store_true", help="Recreate the collection first")
    args = parser.parse_args()

    data = load_processed(paths=SETTINGS.paths)
    movies = data["movies"]

    fetcher = make_tmdb_fetcher(movies, api_key=SETTINGS.tmdb_api_key)
    source = "TMDB overviews" if fetcher else "title+genres"
    enriched = build_movie_text(movies, fetcher)
    print(f"Embedding {len(enriched):,} movies ({source}) ...")

    t0 = time.time()
    vectors = embed_texts(enriched["text"].tolist())
    print(f"Embedded in {time.time() - t0:.1f}s -> {vectors.shape}")

    store = VectorStore(config=SETTINGS.qdrant)
    if args.reset:
        store.reset()
    indexed = store.index_movies(enriched, vectors)
    print(f"Indexed {indexed:,} points into Qdrant collection '{SETTINGS.qdrant.collection}'")
    store.close()


if __name__ == "__main__":
    main()
