# 📈 CineMatch AI — Progress Tracker

> Keep this file updated as you complete each step. Mark with `[x]`.
> Last updated: 07 Aug 2026 — **all core phases complete and verified on MovieLens 100k.**
>
> 📖 For exact commands + copy-paste AI prompts, see **`PROMPTS.md`** (the build playbook).

---

## Phase 0 — Scaffolding & Foundation

- [x] Project structure created (`src/`, `scripts/`, `frontend/`, `tests/`)
- [x] `requirements.txt` written
- [x] `.env.example` written (dataset size, Qdrant, embedding model, Ollama, TMDB)
- [x] `docker-compose.yml` for Qdrant server mode
- [x] Central config module `src/cinematch/config.py`
- [x] Core source modules scaffolded:
  - [x] `data.py` (MovieLens ETL / parquet cache)
  - [x] `text.py` (movie metadata → embedding text + TMDB enrich)
  - [x] `embeddings.py` (sentence-transformers wrapper)
  - [x] `vector_store.py` (Qdrant embedded / remote)
  - [x] `models/` (SVD + item-based CF collaborative, content, hybrid rerank)
  - [x] `feedback.py` (like/dislike/watchlist signals)
  - [x] `explainer.py` (Ollama LLM + rule-based fallback)
  - [x] `recommend.py` (two-stage pipeline orchestrator)
  - [x] `api.py` (FastAPI app)
- [x] CLI scripts scaffolded:
  - [x] `scripts/download_data.py`
  - [x] `scripts/train.py`
  - [x] `scripts/eval.py`
  - [x] `scripts/index_vectors.py`
  - [x] `scripts/run_api.py`
- [x] Streamlit frontend scaffolded (`frontend/streamlit_app.py`)
- [x] Smoke tests written (`tests/test_smoke.py`)
- [x] `README.md` documentation written
- [x] `pyproject.toml` added + `pip install -e .` (package importable everywhere)

## Phase 1 — Environment Setup

- [x] Virtual environment created (`.venv`, Python 3.13.9)
- [x] Dependencies installed (`pip install -r requirements.txt`) — numpy, pandas, scikit-learn,
      sentence-transformers (CPU torch), qdrant-client, fastapi, streamlit, pytest
- [x] `.env` created from `.env.example` (defaults: `MOVIELENS_SIZE=100k`, embedded Qdrant)

## Phase 2 — Data Ingestion (MovieLens)

- [x] Downloaded MovieLens 100k dataset
- [x] Raw files verified (`data/raw/ml-100k/`)
- [x] ETL ran → cached parquet (`data/processed/`)
      → `Ingested 1,682 movies, 100,000 ratings from MovieLens 100K.`

## Phase 3 — Offline Model Training

- [x] Trained SVD collaborative model (`models_cache/svd_100k.pkl`)
      → `Trained in 0.2s: 943 users x 1,682 movies | explained variance 36.93%`
- [x] Added **item-based collaborative filtering** (`ItemBasedCF`) — co-rating cosine similarity,
      used alongside SVD in the hybrid CF path
- [x] Ran offline evaluation
- [x] Logged baseline metrics (see Phase 8 table — protocol now NCF negative-sampling)

## Phase 4 — Vector Search (Semantic)

- [x] Downloaded embedding model `all-MiniLM-L6-v2` from HuggingFace
- [x] Built movie embeddings (1682 × 384-dim, title+genres)
- [x] Indexed into Qdrant embedded mode (`qdrant_storage/`) — 1,682 points
- [x] Smoke-tested semantic search (`/api/movies/search?q=...`)

## Phase 5 — Backend API (FastAPI)

- [x] API running on `http://localhost:8000/docs`
- [x] Tested `GET /api/recommend/{user_id}` (hybrid + explanations)
- [x] Tested `GET /api/movies/search?q=...` (semantic)
- [x] Tested `POST /api/recommend/query` (query blend)
- [x] Tested `POST /api/feedback` (like/dislike/watchlist → live re-rank)
- [x] Tested `GET /api/user/{user_id}/profile`
- [x] Tested `GET /api/movies/{movie_id}`
- [x] Tested `GET /api/explain/{user_id}/{movie_id}`

## Phase 6 — Frontend (Streamlit)

- [x] Streamlit app launches on `http://localhost:8501` (HTTP 200)
- [x] Recommendations for a user render
- [x] Semantic search from UI
- [x] Feedback buttons work (re-rank updates)

## Phase 7 — Extras & Optimizations

- [x] `pytest -q` — **8 passed**
- [ ] Qdrant in Docker server mode (`docker compose up` + `QDRANT_URL=http://localhost:6333`)
- [ ] LLM explanations via Ollama (`ollama pull llama3.2`) — template fallback used until then
- [ ] TMDB enrichment (set `TMDB_API_KEY`) → real plot overviews, re-run `index_vectors.py`
- [ ] Scale up to MovieLens 1m / 25m (`MOVIELENS_SIZE=1m`)
- [ ] Tune weights: `SVD_FACTORS`, `WEIGHT_CF`, `WEIGHT_CB`, `WEIGHT_POP`

## Phase 8 — Metrics Targets

> **Eval protocol note (updated):** offline eval uses the NCF-style negative-sampling protocol
> — leave one rating out per user, then rank the positive item against 99 sampled negatives
> (random chance HR@10 = 0.10). This is the standard protocol used in recommender-systems papers.

| Metric | Target | Achieved (ml-100k) |
| --- | --- | --- |
| Hit Rate@10 (1 pos + 99 neg) | ≥ 0.40 | **0.381** |
| MRR | ≥ 0.25 | **0.202** |
| Candidate retrieval latency | < 250 ms | ~instant (in-memory SVD + itemCF) |
| LLM explanation latency | < 1.5 s | template fallback (Ollama not installed) |

---

## Notes / Learnings

**Changes made while completing the project (07 Aug 2026):**

- **Packaging:** added `pyproject.toml` and ran `pip install -e .` so `cinematch` is importable
  from scripts, tests, uvicorn and streamlit without manual `sys.path` hacks.
- **Eval fixes (`scripts/eval.py`):**
  - Fixed a crash in leave-one-out splitting (`group.iloc[keep]` returned a Series → broken DataFrame).
  - Switched to the standard **negative-sampling protocol** (chance HR@10 = 0.10) with a proper
    rank computation — the old full-corpus ranking made the metric look broken (0.002).
- **Item-based CF (`models/collaborative.py`):** added `ItemBasedCF` (co-rating cosine similarity).
  It is dramatically better than pure SVD for the "predict what the user just watched" task and is
  now blended into the CF candidate path (40% SVD / 60% itemCF).
- **Qdrant 1.19 API:** `client.search(...)` was removed → migrated to `query_points(...)`, and
  flattened query vectors to `(dim,)` (embed_query returned `(1, dim)` → multivector error).
- **Graceful degradation:** if the vector index isn't built yet, the API falls back to CF +
  popularity instead of crashing; `/api/movies/search` returns a clear 503 with instructions.
- **Test fixes:** two smoke tests had incorrect assumptions (popularity equality for distinct random
  draws; zero-delta movies being present in the delta dict). Rewrote them with controlled data.
- **Known cosmetic issue:** qdrant's embedded client prints an `import of msvcrt halted` warning on
  interpreter shutdown — harmless, comes from portalocker during GC.
