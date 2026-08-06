# 🛠️ CineMatch AI — Build-It-Yourself Playbook

> A step-by-step guide to build and run CineMatch yourself in VS Code.
> Each phase has: **what to do**, **exact commands**, **which files matter**, and a **copy-paste AI prompt** you can give to Cursor / Copilot / opencode / Claude when you get stuck or want help with that step.
>
> Tick items off in **`PROGRESS.md`** as you finish them.

---

## Before you start (read this once)

- Everything is **already coded**. Your job is to **run it in the right order**, understand it, and fix what breaks. This is the best way to learn.
- Open the project in VS Code (`File → Open Folder` → select this folder).
- Open a **PowerShell terminal** inside VS Code (`Ctrl + Shift + \``).
- **Always** run commands from the project root (the folder this file is in).
- The virtual environment already exists (`.venv`, Python 3.13.9). You must **activate it** in every new terminal before running `python`/`pip`:

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Golden rule:** run one phase → verify it → tick it off → move on. Never skip ahead.

### Project map (where everything lives)

| Path | What it is |
| --- | --- |
| `src/cinematch/config.py` | All settings (dataset size, weights, model names) |
| `src/cinematch/data.py` | MovieLens download + ETL + parquet cache |
| `src/cinematch/text.py` | Movie metadata → embedding text (+ TMDB enrichment) |
| `src/cinematch/embeddings.py` | sentence-transformers wrapper (384-dim vectors) |
| `src/cinematch/vector_store.py` | Qdrant wrapper (embedded or remote) |
| `src/cinematch/models/collaborative.py` | TruncatedSVD matrix factorization (CF) |
| `src/cinematch/models/content.py` | Profile-vector content scoring (CBF) |
| `src/cinematch/models/hybrid.py` | Weighted fusion + re-ranking |
| `src/cinematch/feedback.py` | Real-time like/dislike/watchlist signals |
| `src/cinematch/explainer.py` | Ollama LLM + rule-based fallback explanations |
| `src/cinematch/recommend.py` | Two-stage pipeline orchestrator |
| `src/cinematch/api.py` | FastAPI app (all endpoints) |
| `scripts/` | `download_data`, `train`, `eval`, `index_vectors`, `run_api` |
| `frontend/streamlit_app.py` | Streamlit UI |
| `tests/test_smoke.py` | Smoke tests |

---

## Phase 1 — Environment Setup ✅ (mostly done)

**Do this:**
1. Activate the venv (command above).
2. Install dependencies (first time only — downloads torch + sentence-transformers, ~2 GB, be patient):

```powershell
pip install -r requirements.txt
```

3. Create your local config:

```powershell
Copy-Item .env.example .env
```

4. Verify install:

```powershell
python -c "import pandas, sklearn, fastapi, qdrant_client, streamlit; print('all good')"
```

**Troubleshooting:** if `pip` fails on a package, update pip first (`python -m pip install --upgrade pip`) and retry.

> **Prompt to paste (if you need help):**
> "I'm setting up the CineMatch movie recommender in VS Code on Windows. The venv at `.venv` uses Python 3.13.9. I ran `pip install -r requirements.txt` and got this error: [paste error]. Fix it and tell me the exact command to verify the install. Do not modify any source code."

---

## Phase 2 — Data Ingestion (MovieLens)

**Do this:**

```powershell
python scripts/download_data.py --size 100k
```

This downloads `ml-100k` (~5 MB) into `data/raw/`, then ETLs and caches cleaned frames to `data/processed/movies.parquet` + `ratings.parquet`.

**Verify:**
- Folder `data/raw/ml-100k/` exists (contains `u.data`, `u.item`).
- Folder `data/processed/` contains `movies.parquet` and `ratings.parquet`.
- Terminal prints something like: `Ingested 1,682 movies, 100,000 ratings from MovieLens 100K.`

**Read these files to understand what just ran:** `scripts/download_data.py` and `src/cinematch/data.py`.

> **Prompt (if stuck):**
> "I ran `python scripts/download_data.py --size 100k` in the CineMatch project and got: [paste error/output]. Diagnose it and give me the fix as a single PowerShell command I can paste. Keep it minimal — do not rewrite the pipeline."

---

## Phase 3 — Train & Evaluate the Collaborative Model (CF)

**Do this:**

```powershell
python scripts/train.py
python scripts/eval.py --topk 10
```

`train.py` fits a **TruncatedSVD** matrix factorization and saves it to `models_cache/`. `eval.py` runs leave-one-out evaluation and prints **Hit Rate@10** and **MRR**.

**Verify:** `models_cache/` contains the saved model; eval prints real numbers (they'll be modest on 100k — that's fine).

**Note:** your target is HitRate@10 ≥ 0.75, MRR ≥ 0.65. You'll tune weights later in `src/cinematch/config.py` / `.env` (`SVD_FACTORS`, `WEIGHT_CF`, `WEIGHT_CB`, `WEIGHT_POP`).

**Read:** `src/cinematch/models/collaborative.py`, `scripts/train.py`, `scripts/eval.py`.

> **Prompt:**
> "After running `python scripts/train.py` and `python scripts/eval.py --topk 10` for the CineMatch project, I got: [paste output]. Are these numbers reasonable for MovieLens 100k? Suggest 3 concrete, small config changes (in `.env` or `src/cinematch/config.py`) that most improve Hit Rate@10 and MRR, and explain why. Only list the exact edits — don't rewrite code."

---

## Phase 4 — Semantic Vector Search (Content Path)

**Do this:**

```powershell
python scripts/index_vectors.py
```

First run downloads the embedding model `all-MiniLM-L6-v2` (~90 MB) from HuggingFace, embeds every movie's `title + year + genres`, and indexes them into **Qdrant** (embedded mode → stored in `qdrant_storage/`).

**Verify:** `qdrant_storage/` exists; the script reports how many vectors were indexed.

**Smoke-test semantic search** (in a Python one-liner or a scratch script `scratch_test.py`):

```python
# scratch_test.py
from cinematch.recommend import hybrid_search_query
print(hybrid_search_query("a gritty 90s thriller with a mind-bending plot twist"))
```

**Read:** `src/cinematch/embeddings.py`, `src/cinematch/vector_store.py`, `src/cinematch/text.py`.

> **Prompt:**
> "I ran `python scripts/index_vectors.py` in CineMatch. Output: [paste]. It [worked / failed with error X]. If it failed, fix it. If it worked, explain in 5 bullet points exactly what the pipeline did — model, text source, vector dimension, Qdrant mode, where data is stored."

---

## Phase 5 — Backend API (FastAPI)

**Do this:**

```powershell
python scripts/run_api.py
```

Open **http://localhost:8000/docs** in your browser — this is Swagger UI with clickable test buttons for every endpoint.

**Test in this order:**
| Endpoint | What it proves |
| --- | --- |
| `GET /api/recommend/{user_id}` | Hybrid recs work end-to-end |
| `GET /api/movies/search?q=gritty 90s thriller` | Semantic search works |
| `POST /api/feedback` | Feedback loop works (like/dislike/watchlist) |
| `GET /api/explain/{user_id}/{movie_id}` | Explanations work |

**Read:** `src/cinematch/api.py`, `src/cinematch/recommend.py`, `src/cinematch/feedback.py`.

> **Prompt:**
> "The CineMatch FastAPI server runs but [endpoint] returns [error/behaviour]. Here is the request and response: [paste]. Fix the bug and give me the exact curl or browser URL to re-test. Don't change unrelated code."

---

## Phase 6 — Frontend (Streamlit)

**Do this** — keep the API running, then in a **second terminal**:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/streamlit_app.py
```

A browser tab opens with the UI. Try: view recommendations for a user, run a semantic search, click like/dislike and watch the re-ranking update.

**Read:** `frontend/streamlit_app.py`.

> **Prompt:**
> "My Streamlit UI at `frontend/streamlit_app.py` [doesn't load / shows this error / the feedback buttons don't re-rank]. Paste: [error or screenshot text]. Debug it and give the exact lines to change."

---

## Phase 7 — Extras & Optimizations (do these last)

**A. Run the smoke tests**

```powershell
pytest -q
```

**B. Qdrant in Docker server mode** (instead of embedded):

```powershell
docker compose up -d
```

Then set `QDRANT_URL=http://localhost:6333` in `.env` and re-run `python scripts/index_vectors.py`.

**C. LLM explanations with Ollama** (local, free):

```powershell
ollama pull llama3.2
```

Ollama is already configured (`OLLAMA_URL`/`OLLAMA_MODEL` in `.env`). If Ollama isn't running, the app falls back to rule-based explanations — it never crashes.

**D. TMDB enrichment** (real plot summaries): get a free key at the TMDB website, set `TMDB_API_KEY` in `.env`, re-run `index_vectors.py`.

**E. Scale up:** set `MOVIELENS_SIZE=1m` in `.env` and repeat Phase 2 → 4.

**F. Tune quality:** try `SVD_FACTORS=100`, `WEIGHT_CF=0.6`, `WEIGHT_CB=0.3`, `WEIGHT_POP=0.1`; re-run `train.py` + `eval.py` and compare HitRate@10 / MRR against the target table in `PROGRESS.md`.

> **Prompt:**
> "CineMatch Phase 7: I did [X] and got [result/error]. Here is the relevant log: [paste]. Tell me the next 2–3 things to try to hit the Phase 8 targets (HitRate@10 ≥ 0.75, MRR ≥ 0.65), starting with the change most likely to move the metric."

---

## Common gotchas

- **`ModuleNotFoundError`** → venv not activated, or you ran `pip` in a different terminal. Activate `.venv` first.
- **Torch takes long / download errors** → normal on first `pip install`. Retry after `python -m pip install --upgrade pip`.
- **HuggingFace model download fails** (no internet / proxy) → check your connection; the model is cached after the first success.
- **Port 8000 already in use** → run the API with `python scripts/run_api.py --port 8001` (check `run_api.py` for the flag).
- **Qdrant "collection not found"** → you ran the API before `index_vectors.py`. Run Phase 4 first.
- **`git` not initialized** → run `git init` once, then commit after each working phase: `git add .` → `git commit -m "phase N"`.

---

## Your first AI prompt template (always works)

> "I am following the CineMatch build playbook in `PROMPTS.md` and just finished Phase [N].
> Command I ran: `[command]`
> Output / error: [paste exactly]
> What I expect: [what should have happened]
> Please: (1) tell me what went wrong in one line, (2) give me the exact command or file edit to fix it, (3) tell me how to verify it worked. Do not rewrite other parts of the project."
