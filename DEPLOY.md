# 🚀 Deploying CineMatch AI

Two supported paths. The **recommended free path** is Streamlit Community Cloud:
the UI runs standalone (engine in-process, no FastAPI/Qdrant server) in a
"light" mode that fits the 1 GB free tier. A second, full-stack Docker path for
Hugging Face Spaces requires a paid plan (see below).

---

## Option 1 — Streamlit Community Cloud (recommended, free)

Streamlit Community Cloud builds from your GitHub repo and runs the app in
**light mode**: numpy vector search over a precomputed index
(`data/processed/light_index.pkl`, 9561×384) + ONNX embeddings (`fastembed`).
No torch, no Qdrant, no separate API server — everything runs in one process.

### 1. Bake the light assets (once)

```powershell
python scripts/build_light_assets.py
```

This writes `data/processed/light_index.pkl` (the semantic index) and ensures
`models_cache/svd_100k.pkl` (the SVD matrix) exists. Both are **committed to
git** (whitelisted in `.gitignore`) so the cloud runtime never trains or
re-downloads them.

### 2. Push to GitHub

```powershell
git add .
git commit -m "deploy: Streamlit Cloud"
git push origin master
```

### 3. Create the app

1. Go to https://share.streamlit.io (or the Deploy tab in GitHub) → **Create app**
2. Repo: `sangameshh31/movie-recommendation-system`, branch `master`
3. **Main file:** `frontend/streamlit_app.py`
4. **Python version:** 3.12 (pinned via `runtime.txt`)
5. Dependencies come from `requirements.txt` (the light runtime)

### 4. Secrets / env (Advanced settings)

| Env var | Value | Why |
| --- | --- | --- |
| `CINEMATCH_LIGHT` | `1` | Selects the light engine path (defaults to this only via env; the cloud app sets it). |
| `TMDB_API_KEY` | your key | Live trailers, cast/crew, people search. Without it those pages degrade gracefully. |

`requirements.txt` installs `streamlit`, `fastembed`, and friends — the first
boot downloads the MiniLM ONNX model (~90 MB) from HuggingFace Hub and caches
it for the life of that deployment.

Note: accounts + feedback are file-backed and ephemeral on Community Cloud
(no persistent disk). That's the accepted trade-off for the free tier.

---

## Option 2 — Hugging Face Spaces Docker (full stack, needs PRO)

The original full-stack deploy (`DEPLOY` docs below this file) runs
FastAPI + Streamlit + embedded Qdrant in one container with everything baked in.
Docker Spaces on the **free** tier are PRO-only as of 2026, so this path needs a
paid plan (or a free-tier upgrade after account maturity/community grant).

If you go this route: the `Dockerfile` installs the full stack via
`requirements-dev.txt` (torch, qdrant-client, fastapi) and the baked pipeline
`scripts/train.py && scripts/index_vectors.py` runs at build time. The
`CINEMATCH_LIGHT` env var must **not** be set there.

---

## Troubleshooting (Streamlit Cloud)

| Symptom | Fix |
| --- | --- |
| App logs show module-not-found | `requirements.txt` must be at the repo root; redeploy after pushing a change. |
| First boot is slow (~90 s) | The ONNX model downloads once per deployment; subsequent reruns use the cache. |
| Memory >1 GB OOM | Ensure `CINEMATCH_LIGHT=1` is set; it drops torch + Qdrant from the process. |
| Trailer / cast pages empty | `TMDB_API_KEY` missing/invalid — add it in Advanced settings and redeploy. |
| Changes not showing | Streamlit rebuilds on every push to the connected branch — push again. |
| GitHub Actions tries to run | No CI configured; if you add one, build only on `main`/`master` and skip light assets. |
