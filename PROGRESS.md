# 📈 CineMatch AI — Progress Tracker

> Keep this file updated as you complete each step. Mark with `[x]`.
> Last updated: 07 Aug 2026 — **7 new features: account library, people pages, surprises, refinement, trailers, onboarding.**
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
- [x] **Indian cinema** — curated catalog (**2,818 TMDB Indian films across 13
      languages** + Dr. Rajkumar filmography, Hindi→Bhojpuri)
      with seeded "critic" ratings; fully searchable, recommendable and
      trendable via `/api/movies/trending?origin=indian&language=Tamil`
- [x] **Full catalog (9,561 titles)** — Hollywood + world cinema, **anime (979)**,
      **TV series (603)** and animation/cartoons, each with `media_type` +
      `origin` markers and IMDb-style `vote_average` badges
- [x] **IMDb-style UI** — dark theme + gold accent, logo band, pill navigation,
      Featured hero, horizontal scroll rails (Popular / Top rated / New /
      Indian / Anime / TV), IMDb rating badges on every poster, explore chips
- [x] **Real TMDB posters** — poster URLs captured inline from the catalog fetch
      passes (no separate pass needed); gradient fallback when missing
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
- [x] **7 new features** (account library, "Because you liked", people pages,
      onboarding, conversational search refinement, trailers + external links,
      Surprise me) — see Notes below

## Phase 8 — Metrics Targets

> **Eval protocol note (updated):** offline eval uses the NCF-style negative-sampling protocol
> — leave one rating out per user, then rank the positive item against 99 sampled negatives
> (random chance HR@10 = 0.10). This is the standard protocol used in recommender-systems papers.

| Metric | Target | Achieved (ml-100k + Indian catalog) |
| --- | --- | --- |
| Hit Rate@10 (1 pos + 99 neg) | ≥ 0.40 | **0.415** |
| MRR | ≥ 0.25 | **0.241** |
| Candidate retrieval latency | < 250 ms | ~instant (in-memory SVD + itemCF) |
| LLM explanation latency | < 1.5 s | template fallback (Ollama not installed) |

---

## Notes / Learnings

**Changes made while completing the project (07 Aug 2026):**

- **Seven new features (this batch):**
  - **Persistent account library (`src/cinematch/feedback.py` + `GET /api/user/{id}/library`):**
    the feedback store is no longer in-memory — likes, dislikes, watchlist,
    **watched** and **star ratings** now persist to `data/feedback.json`
    (thread-safe, atomic writes) and survive restarts. The Profile page shows
    "Your library" rails (Liked / Watchlist / Watched / ⭐ Your ratings).
  - **"Because you liked" (`recommend.py` `_because_of`):** every recommendation
    now carries a `because_of` list (top-2 liked titles by raw vector cosine
    similarity) shown as a caption under each card, e.g. "🎯 Because you liked:
    Inception, The Matrix". Feedback likes now feed the content profile too
    (`liked_movies()` merges MovieLens 4★ with live 👍 signals) and
    rated/watched/liked titles are excluded from future recommendations.
  - **Director & cast exploration (`src/cinematch/people.py` + `GET /api/people/search?q=`):**
    TMDB person search + profile + combined credits (7-day cache in
    `people_cache.parquet`), mapped back onto catalog titles. Director/Producer/
    Writer chips and every cast card on the detail page are clickable
    (`?person=Name`) and open a **People page** with photo, bio, and "in your
    catalog" / "more from this person" rails; "← Back" returns to the movie.
  - **Sign-up onboarding (Profile):** new accounts get a "Pick your favorites"
    multi-select of 40 popular titles; saving marks them 👍 (seeding the taste
    profile immediately).
  - **Conversational search refinement (`POST /api/search/refine`):** after a
    search, an expander lets you "➕ add more like X" and "➖ less of Y" — the
    engine embeds each nudge and re-searches `seed + Σadd - Σremove` so results
    shift toward/away from those concepts.
  - **Trailers + external links (detail page):** TMDB `videos`/`homepage`/
    `imdb_id` are fetched with the detail (`append_to_response=credits,videos`,
    cache bumped to v2) and rendered as Trailer / Official site / IMDb / TMDB
    buttons.
  - **Surprise me (Home):** one-click personalized pick — content profile →
    80 vector candidates → random draw weighted by similarity (popularity pool
    for cold users), with 👍 / ➕ / "Roll again".
  - Verified: all endpoints exercised, `pytest -q` → **9 passed**, AppTest
    sweep over all 8 pages + every new feature → **0 exceptions**.

- **Movie detail pages (`src/cinematch/details.py` + `/api/movies/{id}/details`):**
  clicking any poster tile (or the "Details" button on result cards) now opens a
  full IMDb-style detail view — synopsis, tagline, runtime, release date, status,
  genre chips, **cast strip** (photos + characters), **Director / Producer / Writer**
  credits, a TMDB rating and a "More like this" rail (vector neighbors, genre
  fallback). Details are fetched **lazily from TMDB on first open** (`/movie|tv/{id}`
  with `append_to_response=credits`, or `/search` by title+year for MovieLens rows
  without a `tmdb_id`) and cached to `data/processed/details_cache.parquet`
  (30-day TTL) so re-opens are instant. Every fetch is best-effort — failures fall
  back to the local catalog. Tiles deep-link via `?detail=<id>` (URL shareable);
  nav pills / "Back" close the view.
- **Sign-in arranged like a real product:** the auth form moved out of the sidebar
  into a centered **auth card on the Profile page** (sign in / create account in two
  taps, guest status shown). The header now shows a compact **account pill**
  ("🔐 Sign in" or "👤 name") that jumps to Profile; the sidebar shows a minimal
  signed-in status with a sign-out button. Accounts live in `data/users.json`
  (gitignored), no backend changes needed.
- **Catalog scale-up to 9,561 titles (`scripts/fetch_catalog.py` v2 + `scripts/migrate_catalog.py`):**
  the catalog now covers **Hollywood + world cinema, Indian cinema (13 languages),
  anime, animation/cartoons and TV series**:
  - all-time popular movies (150 pages) + top-rated movies + 2024–2026 releases,
  - **2,818 Indian** titles across 13 languages (Hindi, Tamil, Telugu, Malayalam,
    Kannada, Marathi, Bengali, Punjabi, Gujarati, Assamese, Oriya, Urdu, Bhojpuri)
    incl. the Dr. Rajkumar filmography,
  - **979 anime** titles (Japanese animation movies + series),
  - **603 TV series** (top-rated + popular),
  - every row carries `media_type` (`movie`/`series`) + `origin`
    (`indian`/`anime`/`series`/`new`) and a TMDB `vote_average` (0-10) so the UI
    can show IMDb-style rating badges and separate rails.
  - cache `data/processed/tmdb_catalog.parquet` flushes every 200 rows → fully resumable.
- **IMDb-style UI rewrite (`frontend/streamlit_app.py` v2):** dark IMDb look with the
  gold `#f5c518` accent — top logo band, pill navigation (Home / For You / Indian /
  Anime / TV / Search / Profile), a "Featured" hero card, and **horizontal scroll rails**
  (Popular now, Top rated, New releases, Indian, Anime, TV series) of compact poster
  tiles with **gold IMDb-style rating badges**, type tags and hover lift. Real search
  is one click away with explore chips and natural-language queries. All glitter/
  conic-border effects removed in favour of a clean, professional layout.
- **Trending API upgrades:** `/api/movies/trending` now supports `sort_by`
  (`popularity` | `rating` | `new`) and `media_type` filters; `/health` reports
  `indian_movies`, `anime` and `series` counts; embeddings tag TV series so
  queries like "anime series" or "crime tv show" match correctly.
- **TMDB catalog expansion (`scripts/fetch_catalog.py` + `src/cinematch/tmdb_catalog.py`):**
  the catalog grew from 2,061 → **3,442 movies** via TMDB discover (top 250 per year for
  2024/2025/2026 incl. 301 releases from 2026, top 60 per Indian language, and the full
  Dr. Rajkumar filmography from TMDB person 1128070). Final split: **890 Indian** films
  (Kannada 155, Malayalam 91, Telugu 89, Tamil 82, Hindi 65, Bengali 25, Punjabi 2,
  Marathi 2) + 2,552 recent/world cinema. Source data cached in
  `data/processed/tmdb_catalog.parquet` (resumable, deduped by `tmdb_id` + normalized title).
- **TMDB network fix (`src/cinematch/tmdb_net.py`):** the ISP resolver hands out a dead edge
  IP for `api.themoviedb.org` (TCP timeout). `patch_tmdb_dns()` resolves the real CloudFront
  IPs via DNS-over-HTTPS, TCP-preflights them, and pins them through a `socket.getaddrinfo`
  monkeypatch. All TMDB calls (catalog fetch, poster fetch, text enrich) go through it.
- **Popularity-aware critic seeding (`tmdb_catalog.seed_movie_ratings`):** seeded ratings for
  TMDB titles now scale with quality (TMDB vote average) *and* popularity (vote count — how many
  critics have "seen" each film). Trending is no longer dominated by obscure classics; it now
  surfaces real hits (e.g. 2026 releases, Gangubai Kathiawadi, Koi... Mil Gaya) alongside
  Dr. Rajkumar classics. Ratings rebuilt to 140,953; SVD retrained (3,442 movies,
  explained variance 30.21%); eval improved to **HR@10 0.415 / MRR 0.241**.
- **Indian cinema catalog (`src/cinematch/indian_cinema.py`):** MovieLens has no
  Indian films, so we appended 379 hand-curated titles across 8 languages with a
  `language` + `origin` marker (later expanded to 890 Indian titles via the TMDB
  augmentation). Synthetic "critic" users seed ratings so the
  collaborative paths (popularity, item-CF) score them too. Augmentation is
  idempotent and wired into `load_processed`, so `train`, `index_vectors`,
  `eval` and the API all see the combined catalog (3,442 movies total).
- **Language-aware embeddings:** embedding text now includes the film's language
  ("... :: Tamil film"), so queries like *"a Malayalam crime thriller"* or
  *"Hindi romantic drama"* match correctly.
- **New API endpoints:** `GET /api/movies/trending?origin=indian&language=Tamil`
  and optional `origin`/`language` filters on `GET /api/movies/search`.
- **UI overhaul:** dark cinematic theme with animated effects (aurora backdrop, shimmer title,
  twinkling stars, rotating conic-gradient poster borders), deterministic gradient poster cards
  (no image CDN needed), genre/language chips, star ratings, one-click Explore
  prompts, a dedicated Indian Cinema tab with trending + mood search, and a "Filter results"
  panel (genre / year range / min-match score) on every grid.
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
