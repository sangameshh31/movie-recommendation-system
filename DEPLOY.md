# 🚀 Deploying CineMatch AI to Hugging Face Spaces

Free, permanent hosting (no spin-down) for a live demo of the full stack:
FastAPI + Streamlit + Qdrant (embedded) in **one Docker container**, with the
dataset, model, and vectors baked in at build time so the app starts instantly.

> No Docker needed on your machine — Hugging Face builds the image in the cloud.
> Expected build time: a few minutes (installs torch CPU + downloads the
> embedding model once, then trains the SVD model and indexes ~9.5k titles).

---

## 1. Push the repo to GitHub

A `git` remote already points at
`https://github.com/sangameshh31/movie-recommendation-system` (branch `master`).
Commit the latest work and push:

```powershell
git add .
git commit -m "CineMatch AI: complete hybrid recommender"
git push origin master
```

The processed catalog **is** committed on purpose (`.gitignore` whitelists
`data/processed/{movies,ratings,tmdb_catalog}.parquet`) — the Docker build
copies it in, so no MovieLens/TMDB downloads happen at build time. Everything
else (`data/`, `.env`, `models_cache/`, `qdrant_storage/`) stays gitignored.

## 2. Create the Space

1. Go to https://huggingface.co/new-space
2. **Space name:** `cinematch-ai` (anything you like)
3. **License:** MIT (matches the repo)
4. **SDK:** `Docker`  ← important
5. **Hardware:** keep **CPU basic** (free)
6. Create.

## 3. Connect the code

Choose **one** of these:

**Option A — sync from GitHub (easiest)**
1. In the Space page: `Settings → Repo management → Connect GitHub repo`
2. Pick your repo and branch (`master`). Every `git push` to GitHub triggers a rebuild.

**Option B — push directly to the Space**
```powershell
git remote add hf https://USERNAME:HF_TOKEN@huggingface.co/spaces/USERNAME/cinematch-ai
git push hf master --force
```
Get `HF_TOKEN` from https://huggingface.co/settings/tokens (a "write" token).
> The Space builds from the tip of its main branch, so push everything including
> the `Dockerfile`, `start.sh`, `README.md` (with the `sdk: docker` header), the
> `src/` / `scripts/` / `frontend/` folders, and the three `data/processed/*.parquet`
> files.

## 4. Set the Space secrets

`Settings → Variables and secrets`:

| Secret | Value | Why |
| --- | --- | --- |
| `TMDB_API_KEY` | your key from https://www.themoviedb.org/settings/api | Enables live trailers, cast/crew, and people search. Without it those pages degrade gracefully. |
| `DATA_DIR` | `/data` | Points the app at the persistent volume so accounts + feedback survive restarts (enable storage first, step 5). |

## 5. Enable persistent storage (optional but recommended)

`Settings → Persistent Storage → Enable` (free tier gives 20GB).
Combined with `DATA_DIR=/data`, this makes `users.json` (accounts), `feedback.json`
(likes / dislikes / watchlist / watched / stars) and the detail caches survive
container restarts and rebuilds.

## 6. Wait for the build

The Space shows `Building` while the Dockerfile runs. It builds in this order:

```
pip install deps        # torch CPU, fastapi, streamlit, qdrant-client ...
copy baked catalog      # movies + ratings + tmdb_catalog parquet (~3 MB)
train SVD model         # on the baked ratings (290k rows)
index ~9.5k embeddings  # MiniLM-L6-v2 → embedded Qdrant
start API (:8000)       # internal
start Streamlit (:7860) # the public port
```

## 7. Use it

Open **https://USERNAME-cinematch-ai.hf.space**

- Recommendations + "why" explanations (because-of)
- Natural-language search (`/api/movies/search`)
- Like / Dislike / Watchlist / Watched / star ratings
- Live trailers + cast/crew (with `TMDB_API_KEY`)
- Surprise me, people search, profile library rails

Note: only the Streamlit port is exposed to the internet; the FastAPI backend
runs on the container-internal port `8000`, which the UI calls directly.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Build fails on `pip install -r requirements.txt` | Free Space CPU quota can be slow; just retry — layers are cached. |
| UI loads but search errors | Shouldn't happen (vectors baked in). If seen, check Space logs: `Settings → View logs` for the `index_vectors` RUN output. |
| "Connection refused" to Ollama | Expected — no Ollama in the container; the rule-based explainer is used automatically. |
| Trailer / cast pages empty | `TMDB_API_KEY` not set or invalid — set the Space secret and rebuild. |
| Accounts / feedback reset on restart | Enable Persistent Storage and set `DATA_DIR=/data` (steps 4–5). |
| Changes not appearing | Rebuilds only trigger on push to the connected branch. `git push` again. |

## Optional: run the same container locally

Install Docker Desktop, then:

```powershell
docker build -t cinematch .
docker run -p 7860:7860 cinematch
```

→ http://localhost:7860 (Streamlit) and http://localhost:8000/docs (API).
