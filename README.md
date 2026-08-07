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
- **Feedback loop:** in-memory like/dislike/watchlist signals that re-rank on every request.
- **Indian cinema:** a curated catalog of **379 films across 8 languages**
  (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi) with
  language-aware embeddings — search *"a Malayalam crime thriller"* or browse
  trending titles per language.

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

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/recommend/{user_id}` | Hybrid recommendations for a user |
| POST | `/api/recommend/query` | Hybrid recs blended with a natural-language query |
| GET | `/api/movies/search?q=...` | Semantic / natural-language search (`origin`, `language` filters) |
| GET | `/api/movies/trending?origin=indian` | Top movies by popularity (Indian cinema rail) |
| POST | `/api/feedback` | Record `like` / `dislike` / `watchlist` / `remove` |
| GET | `/api/user/{user_id}/profile` | User's liked movies |
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
├── indian_cinema.py   # curated Indian catalog (379 films) + seeded ratings
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
