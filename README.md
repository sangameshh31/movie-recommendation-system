---
title: CineMatch AI
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎬 CineMatch AI

> **Live app:** https://cine-match-ai-01.streamlit.app

Next-gen hybrid movie recommendation engine: **Collaborative Filtering (SVD + item-based CF)** + **semantic content filtering (sentence embeddings)** + **LLM explanations**, wrapped in a FastAPI backend and a Streamlit UI.

## Architecture

```
 Streamlit UI ──▶ FastAPI API ──▶ Hybrid Recommender
                                     ├─ Stage 1: Candidate generation
                                     │    ├─ CF path   → SVD + item-based CF (co-rating cosine)
                                     │    └─ Content   → Qdrant vector search
                                     ├─ Stage 2: Re-ranking
                                     │    └─ Weighted blend CF + content + popularity + feedback
                                     └─ Explainer (Ollama LLM / rule-based fallback)
```

- **Vector store:** Qdrant. Runs embedded (zero-setup, persisted in `qdrant_storage/`) by default; point `QDRANT_URL` at `http://localhost:6333` (Docker) for the server mode.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- **Feedback loop:** per-account like/dislike/watchlist/watched/star signals
  (persisted to `data/feedback.json`) that re-rank on every request.
- **Indian cinema:** a TMDB-expanded catalog — **2,818 Indian films across 13 languages**
  (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi, Gujarati,
  Assamese, Oriya, Urdu, Bhojpuri — incl. the full Dr. Rajkumar filmography).
- **Everything else:** **9,561 titles total** — Hollywood + world cinema, **979 anime**
  (movies + series), **603 TV series** and animation/cartoons, each with a `media_type`
  and `origin` marker and an IMDb-style rating badge. Language-aware embeddings —
  search *"a Malayalam crime thriller"*, *"anime adventure"* or *"binge-worthy crime
  tv series"* — and browse trending titles per rail. Real TMDB poster art with
  gradient fallback.
- **Movie detail pages:** click any poster to open a full IMDb-style view — plot
  synopsis, tagline, runtime, genre chips, **cast strip**, **Director / Producer /
  Writer** credits, TMDB rating and a **"More like this"** rail. Details are fetched
  lazily from TMDB on first open and cached (`data/processed/details_cache.parquet`);
  shareable URLs via `?detail=<id>`.
- **Accounts:** sign in / create an account from a clean card on the **Profile** page
  (header shows a "🔐 Sign in" / "👤 name" pill); likes, watchlist, **watched**,
  **star ratings** and preferences persist per account (`data/users.json` + `data/feedback.json`).
  New accounts get a **"Pick your favorites"** onboarding screen that seeds the taste profile.
- **Because you liked:** every recommendation shows which of your liked titles it's
  most similar to (raw vector similarity), e.g. "🎯 Because you liked: Inception, The Matrix".
- **Director & cast exploration:** every cast card and Director/Producer/Writer chip on a
  movie page is clickable — it opens a **People page** (photo, bio, their films in the catalog)
  fetched from TMDB and cached (`data/processed/people_cache.parquet`).
- **Trailers & external links:** movie pages fetch TMDB videos + homepage + IMDb id and
  render Trailer / Official site / IMDb / TMDB buttons.
- **Conversational search refinement:** after any search, use "➕ add more like X" /
  "➖ less of Y" to nudge the results (vector additions/subtractions, `POST /api/search/refine`).
- **Surprise me:** a one-click personalized pick on Home (weighted-random draw from your
  taste profile) with 👍 / ➕ / "Roll again".

## Quickstart

```powershell
# 1. Create a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (torch CPU wheel downloads automatically)
pip install -r requirements.txt
pip install -e .            # makes the `cinematch` package importable anywhere

# 3. Copy env config (optional tweaks)
Copy-Item .env.example .env

# 4. Fetch MovieLens 100k and cache cleaned frames
python scripts/download_data.py --size 100k

# 5. (Optional) train the SVD model & evaluate offline
python scripts/train.py
python scripts/eval.py

# 6. Embed movies and index them into Qdrant
python scripts/index_vectors.py

# 7. Start the API
python scripts/run_api.py            # → http://localhost:8000/docs

# 8. In another terminal, start the UI
streamlit run frontend/streamlit_app.py
```

> First run downloads the embedding model (~90 MB) from HuggingFace Hub.

### Deploy free — Streamlit Community Cloud

The UI runs standalone (engine in-process, no FastAPI/Qdrant server) in a
"light" mode that stays under the 1 GB free tier: numpy vector search over a
precomputed index (`data/processed/light_index.pkl`) + ONNX embeddings
(`fastembed`, no torch).

```powershell
python scripts/build_light_assets.py   # precompute light_index.pkl (once)
git add .
git commit -m "deploy: Streamlit Cloud"
git push origin master
```

Then at https://share.streamlit.io: **Create app → from GitHub** →
`sangameshh31/movie-recommendation-system`, main file
`frontend/streamlit_app.py`, Python 3.12 (`runtime.txt`). Advanced settings:
env var `CINEMATCH_LIGHT=1`, optional `TMDB_API_KEY` (live trailers/cast).
Accounts + feedback are ephemeral per deployment (file-backed, not a DB).

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/recommend/{user_id}` | Hybrid recommendations for a user (with `because_of`) |
| POST | `/api/recommend/query` | Hybrid recs blended with a natural-language query |
| GET | `/api/movies/search?q=...` | Semantic / natural-language search (`origin`, `language` filters) |
| GET | `/api/movies/trending?origin=indian` | Top movies by popularity (Indian cinema rail) |
| POST | `/api/feedback` | Record `like` / `dislike` / `watchlist` / `watched` / `rate` (with `value`) / `remove` |
| GET | `/api/user/{user_id}/profile` | User's liked movies |
| GET | `/api/user/{user_id}/library` | Persistent library: liked / watchlist / watched / star ratings |
| GET | `/api/people/search?q=...` | Director/actor profile + their credits in the catalog |
| POST | `/api/search/refine` | Conversational refinement (`seed` + `additions`/`removals` vectors) |
| GET | `/api/surprise?user_id=...` | One random personalized pick |
| GET | `/api/movies/{movie_id}/details` | Full detail (plot, cast, crew, trailer, links) + similar titles |
| GET | `/api/explain/{user_id}/{movie_id}` | "Why you might like this" |

### LLM explanations (optional)

The explainer calls a local [Ollama](https://ollama.com) server when reachable
(`OLLAMA_URL` / `OLLAMA_MODEL` in `.env`, e.g. `ollama pull llama3.2`).
If Ollama is down, a rule-based template explainer is used — the API never fails.

### TMDB enrichment (optional)

MovieLens has no plot summaries, so content embeddings are built from
`title + year + genres`. Set `TMDB_API_KEY` in `.env` to enrich the embedding
text with real plot overviews fetched from TMDB (rate-limited; only called
during `index_vectors.py`).

## Project layout

```
src/cinematch/
├── data.py            # MovieLens download / ETL / parquet cache
├── indian_cinema.py   # curated Indian catalog + seeded ratings
├── tmdb_catalog.py    # TMDB catalog augmentation (Indian/anime/series/recent)
├── tmdb_net.py        # DNS-over-HTTPS fix for TMDB API timeouts
├── text.py            # movie metadata → embedding text (+ TMDB enrich)
├── embeddings.py      # sentence-transformers wrapper
├── vector_store.py    # Qdrant wrapper (embedded or remote)
├── models/
│   ├── collaborative.py   # TruncatedSVD + ItemBasedCF matrix factorization
│   ├── content.py         # profile-vector content scoring
│   └── hybrid.py          # weighted fusion + re-ranking
├── feedback.py        # real-time like/dislike/watchlist signals
├── explainer.py       # Ollama LLM + rule-based fallback
├── recommend.py       # two-stage pipeline orchestrator
└── api.py             # FastAPI app
frontend/streamlit_app.py
scripts/{download_data,train,index_vectors,eval,run_api}.py
tests/test_smoke.py
```

## Evaluation

`scripts/eval.py` implements the standard **NCF-style negative-sampling protocol**:
leave one rating out per user, then rank the positive item against 99 uniformly
sampled negatives (random chance **HR@10 = 0.10**). Last run on MovieLens 100k:

| Metric | Result |
| --- | --- |
| Hit Rate@10 | **0.395** |
| Mean Reciprocal Rank | **0.216** |

```
python scripts/eval.py --topk 10
```

Tune model quality in `src/cinematch/config.py` / `.env` (`SVD_FACTORS`,
`WEIGHT_CF`, `WEIGHT_CB`, `WEIGHT_POP`, dataset size `1m`/`25m`).
