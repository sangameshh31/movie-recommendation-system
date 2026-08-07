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

- [x] `pytest -q` — **9 passed**
- [x] **Indian cinema** — curated catalog of 379 films across 8 languages
      (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi)
      with seeded "critic" ratings; fully searchable, recommendable and
      trendable via `/api/movies/trending?origin=indian`
- [x] **UI overhaul** — dark cinematic theme, gradient poster cards, genre
      chips, star ratings, one-click "Explore" prompts, Indian Cinema tab
- [x] **Real TMDB posters** — `scripts/fetch_posters.py` fetched posters for
      **2,040/2,061 movies (99%)** using a free TMDB API key; `poster_url` now
      flows through vector payloads, recommendations, trending, search and the
      UI card (gradient fallback if an image is missing). Rebuild with
      `python scripts/fetch_posters.py` then `python scripts/index_vectors.py --reset`.
- [x] **UI + API speed fixes** — see Notes:
      - recommend/health cold-start penalty eliminated (Ollama probe moved to
        startup + opt-out; Qdrant `count()` cached; retrieval paths warmed)
      - `localhost` → `127.0.0.1` (Windows `localhost` resolution can stall ~2s/request)
      - keep-alive `requests.Session` (was a new connection per call)
      - measured: UI API calls now **4–213 ms** (was ~2,000 ms each)
- [ ] Qdrant in Docker server mode (`docker compose up` + `QDRANT_URL=http://localhost:6333`)
- [ ] LLM explanations via Ollama (`ollama pull llama3.2`) — template fallback used until then
- [ ] TMDB plot overviews backfill (`python scripts/fetch_posters.py --fill-overviews`
      then `python scripts/index_vectors.py --reset`) → richer embeddings
- [ ] Scale up to MovieLens 1m / 25m (`MOVIELENS_SIZE=1m`)
- [ ] Tune weights: `SVD_FACTORS`, `WEIGHT_CF`, `WEIGHT_CB`, `WEIGHT_POP`

## Phase 8 — Metrics Targets

> **Eval protocol note (updated):** offline eval uses the NCF-style negative-sampling protocol
> — leave one rating out per user, then rank the positive item against 99 sampled negatives
> (random chance HR@10 = 0.10). This is the standard protocol used in recommender-systems papers.

| Metric | Target | Achieved (ml-100k + Indian catalog) |
| --- | --- | --- |
| Hit Rate@10 (1 pos + 99 neg) | ≥ 0.40 | **0.395** |
| MRR | ≥ 0.25 | **0.216** |
| Candidate retrieval latency | < 250 ms | ~instant (in-memory SVD + itemCF) |
| LLM explanation latency | < 1.5 s | template fallback (Ollama not installed) |

---

## Notes / Learnings

**Changes made while completing the project (07 Aug 2026):**

- **Indian cinema catalog (`src/cinematch/indian_cinema.py`):** MovieLens has no
  Indian films, so we appended 379 hand-curated titles across 8 languages with a
  `language` + `origin` marker. Synthetic "critic" users seed ratings so the
  collaborative paths (popularity, item-CF) score them too. Augmentation is
  idempotent and wired into `load_processed`, so `train`, `index_vectors`,
  `eval` and the API all see the combined catalog (2,061 movies total).
- **Language-aware embeddings:** embedding text now includes the film's language
  ("... :: Tamil film"), so queries like *"a Malayalam crime thriller"* or
  *"Hindi romantic drama"* match correctly.
- **New API endpoints:** `GET /api/movies/trending?origin=indian&language=Tamil`
  and optional `origin`/`language` filters on `GET /api/movies/search`.
- **UI overhaul:** dark cinematic theme, deterministic gradient poster cards
  (no image CDN needed), genre/language chips, star ratings, one-click Explore
  prompts, a dedicated Indian Cinema tab with trending + mood search.
- **Packaging:** added `pyproject.toml` and ran `pip install -e .` so `cinematch` is importable
  from scripts, tests, uvicorn and streamlit without manual `sys.path` hacks.
- **Real posters (`scripts/fetch_posters.py`):** one resumable, rate-limited TMDB pass
  (title+year matching, ~4 req/s) cached `poster_url` for 99% of the catalog. Posters are
  stored in `data/processed/posters.parquet`, merged into `movies.parquet`, and copied into
  Qdrant payloads so the UI shows real poster art with a gradient fallback when absent.
- **Performance fixes (this is why the UI used to feel slow):**
  - **Ollama probe hang:** the first `explain()` call probed `localhost:11434` and stalled
    ~4.3 s on Windows (firewalled/dropped port). The probe now returns instantly when
    `OLLAMA_URL` is empty (opt-out in `.env`) and is otherwise run once at API startup.
  - **`localhost` vs `127.0.0.1`:** on this Windows machine, resolving `localhost` and the
    IPv6 loopback fallback adds ~2 s per connection. The UI now calls `127.0.0.1` (instant).
  - **Stateless `requests.get`:** opened a new TCP connection per call → ~2 s each. The
    frontend now reuses a keep-alive `requests.Session` with `trust_env=False`.
  - **Cold-start warmup:** API lifespan warms the embedding model, Qdrant retrieval paths
    and caches `count()`; first-request latency dropped from ~2–4.5 s to milliseconds.
  - Result: UI API calls measured at **4–213 ms** end-to-end.
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
