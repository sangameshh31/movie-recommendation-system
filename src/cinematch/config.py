"""Central application configuration.

Values are read from environment variables (see ``.env.example``) so the same
codebase works against small local datasets and a full 25M deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels above this module)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(key: str, default: str) -> Path:
    value = os.getenv(key, default)
    p = Path(value).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Paths:
    """All on-disk locations used by the pipeline."""

    root: Path = PROJECT_ROOT
    data_dir: Path = _path("DATA_DIR", "data")
    raw_dir: Path = _path("RAW_DATA_DIR", "data/raw")
    processed_dir: Path = _path("PROCESSED_DATA_DIR", "data/processed")
    models_dir: Path = _path("MODELS_DIR", "models_cache")
    qdrant_dir: Path = _path("QDRANT_DIR", "qdrant_storage")


@dataclass(frozen=True)
class DataConfig:
    """Dataset and ingestion settings."""

    size: str = os.getenv("MOVIELENS_SIZE", "100k").lower()
    test_size: float = 0.2
    random_seed: int = 42
    min_ratings_per_user: int = 1


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = int(os.getenv("EMBEDDING_DIM", "384"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")


@dataclass(frozen=True)
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL", "")
    api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection: str = os.getenv("QDRANT_COLLECTION", "movies")

    @property
    def local_mode(self) -> bool:
        """True when no remote URL is configured (embedded local storage)."""
        return not self.url.strip()


@dataclass(frozen=True)
class RetrievalConfig:
    """Two-stage pipeline settings."""

    n_candidates: int = int(os.getenv("N_CANDIDATES", "150"))
    n_recs: int = int(os.getenv("N_RECS", "10"))
    svd_factors: int = int(os.getenv("SVD_FACTORS", "50"))
    weight_cf: float = float(os.getenv("WEIGHT_CF", "0.5"))
    weight_cb: float = float(os.getenv("WEIGHT_CB", "0.35"))
    weight_pop: float = float(os.getenv("WEIGHT_POP", "0.15"))
    feedback_boost: float = float(os.getenv("FEEDBACK_BOOST", "0.35"))


@dataclass(frozen=True)
class ExplainerConfig:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT", "15"))


@dataclass(frozen=True)
class Settings:
    """Aggregate settings object."""

    paths: Paths = Paths()
    data: DataConfig = DataConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    explainer: ExplainerConfig = ExplainerConfig()
    tmdb_api_key: str = os.getenv("TMDB_API_KEY", "")


SETTINGS = Settings()
