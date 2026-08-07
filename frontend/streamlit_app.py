"""CineMatch AI — Streamlit frontend.

Dark cinematic UI on top of the FastAPI backend. Tabs:

* **For You**      — hybrid recommendations + "why" explanations + feedback
* **Indian Cinema**— trending Indian films by language + mood-based Indian search
* **Search**       — natural-language semantic search over the whole catalog
* **Profile**      — the user's liked movies and watchlist

Run:  streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import os

import requests
import streamlit as st

# 127.0.0.1 (not `localhost`) — on Windows, resolving `localhost` can stall
# ~2s per connection (IPv6 loopback fallback); the numeric address is instant.
API_URL = os.getenv("CINEMATCH_API", "http://127.0.0.1:8000")
DEFAULT_USER = int(os.getenv("CINEMATCH_USER", "1"))

# Reuse one keep-alive connection and skip proxy/WPAD detection so every API
# call after the first is sub-50ms instead of paying ~2s of per-request setup.
_http = requests.Session()
_http.trust_env = False

INDIAN_LANGUAGES = [
    "All",
    "Hindi",
    "Tamil",
    "Telugu",
    "Malayalam",
    "Kannada",
    "Marathi",
    "Bengali",
    "Punjabi",
]

# A curated palette used to generate deterministic "poster" gradients.
PALETTE = [
    ("#7c3aed", "#312e81"),  # indigo
    ("#db2777", "#831843"),  # pink
    ("#2563eb", "#1e3a8a"),  # blue
    ("#0ea5e9", "#0c4a6e"),  # sky
    ("#059669", "#064e3b"),  # emerald
    ("#d97706", "#78350f"),  # amber
    ("#e11d48", "#881337"),  # rose
    ("#4f46e5", "#3730a3"),  # violet
]

_CSS = """
<style>
:root {
  --bg: #0b0f1a;
  --panel: #121a2b;
  --panel2: #0f1626;
  --border: #232f47;
  --text: #e6ecf7;
  --muted: #8fa3c0;
  --accent: #7c3aed;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: var(--panel2);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: var(--text); }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.45rem 1.1rem;
  color: var(--muted);
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #7c3aed, #4f46e5);
  color: white;
  border-color: transparent;
}
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div {
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 10px;
}
.stButton > button {
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 10px;
  transition: all .15s ease;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #7c3aed, #4f46e5);
  border-color: transparent;
  color: white;
}
.cm-hero {
  text-align: center;
  padding: 0.6rem 0 0.2rem 0;
}
.cm-hero h1 {
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(90deg, #c4b5fd, #818cf8, #38bdf8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.2rem;
}
.cm-hero p { color: var(--muted); margin-top: 0; }
.cm-chip {
  display: inline-block;
  background: rgba(124, 58, 237, 0.18);
  border: 1px solid rgba(124, 58, 237, 0.45);
  color: #c4b5fd;
  border-radius: 999px;
  padding: 2px 10px;
  margin: 2px 4px 2px 0;
  font-size: 0.78rem;
  white-space: nowrap;
}
.cm-lang {
  display: inline-block;
  background: rgba(14, 165, 233, 0.15);
  border: 1px solid rgba(14, 165, 233, 0.4);
  color: #7dd3fc;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 0.75rem;
}
.cm-stars { color: #fbbf24; letter-spacing: 1px; font-size: 0.85rem; }
.cm-score { color: var(--muted); font-size: 0.8rem; }
.cm-empty {
  text-align: center; color: var(--muted);
  padding: 3rem 0; font-size: 1.05rem;
}
.cm-section {
  color: #c4b5fd; font-weight: 600; margin: 1.4rem 0 0.4rem 0;
  border-left: 3px solid #7c3aed; padding-left: 0.6rem;
}
</style>
"""


def _color_for(text: str) -> tuple[str, str]:
    digest = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return PALETTE[digest % len(PALETTE)]


def _stars(score: float) -> str:
    n = max(0, min(5, round(score * 5)))
    return "★" * n + "☆" * (5 - n)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _poster_html(movie: dict, height: int = 220) -> str:
    """Poster card: real TMDB image when available, gradient fallback otherwise."""
    title = _escape(movie.get("clean_title") or movie.get("title") or "Untitled")
    year = movie.get("year")
    genres = ", ".join(movie.get("genres", [])[:3]) or "Movie"
    c1, c2 = _color_for(title)
    initial = (movie.get("clean_title") or movie.get("title") or "?").strip()[:1].upper()
    subtitle = (f"{year}" if year else "") + (f"&nbsp;&nbsp;·&nbsp;&nbsp;{_escape(genres)}" if genres else "")
    poster = movie.get("poster_url")

    img = ""
    if poster:
        img = (
            f'<img src="{_escape(str(poster))}" alt="{title}" '
            f'style="width:100%; height:100%; object-fit:cover; object-position:center top; '
            f'display:block; position:absolute; inset:0;" '
            f'onerror="this.remove();var g=this.nextElementSibling;if(g)g.style.display=\'block\';">'
        )
        fallback = (
            f'<div style="display:none; position:absolute; inset:0; '
            f'background:linear-gradient(160deg, {c1} 0%, {c2} 100%);"></div>'
        )
    else:
        fallback = (
            f'<div style="position:absolute; inset:0; '
            f'background:linear-gradient(160deg, {c1} 0%, {c2} 100%);">'
            f'<div style="position:absolute; top:-30px; right:-8px; font-size:{height * 0.85}px; '
            f'font-weight:900; color:rgba(255,255,255,.12); line-height:1;">{initial}</div>'
            f"</div>"
        )

    return f"""
    <div style="border-radius:14px; overflow:hidden; position:relative; height:{height}px;
                margin-bottom:8px; border:1px solid var(--border);
                box-shadow:0 6px 18px rgba(0,0,0,.45);">
      {img}
      {fallback}
      <div style="position:absolute; bottom:0; left:0; right:0; padding:18px 14px 10px 14px;
                  background:linear-gradient(180deg, transparent, rgba(0,0,0,.88));">
        <div style="color:white; font-weight:700; font-size:1.02rem; line-height:1.25; text-shadow:0 1px 3px rgba(0,0,0,.6);">{title}</div>
        <div style="color:rgba(255,255,255,.85); font-size:.8rem; margin-top:3px;">{subtitle}</div>
      </div>
    </div>
    """


def _movie_meta_html(movie: dict) -> str:
    parts = []
    language = movie.get("language")
    if language:
        parts.append(f'<span class="cm-lang">{_escape(language)}</span>')
    for genre in movie.get("genres", [])[:4]:
        parts.append(f'<span class="cm-chip">{_escape(genre)}</span>')
    score = movie.get("score")
    if isinstance(score, (int, float)):
        parts.append(
            f'<span class="cm-stars">{_stars(score)}</span> '
            f'<span class="cm-score">match {score:.0%}</span>'
        )
    return '<div style="line-height:1.9;">' + " ".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict:
    resp = _http.get(f"{API_URL}{path}", params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = _http.post(f"{API_URL}{path}", json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _post_feedback(action: str, movie_id: int) -> None:
    try:
        _post("/api/feedback", {
            "user_id": st.session_state.user_id,
            "movie_id": movie_id,
            "action": action,
        })
        st.toast(f"{action.capitalize()} recorded — recommendations updated", icon="✅")
    except requests.RequestException as exc:
        st.error(f"Feedback failed: {exc}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_card(movie: dict, key: str) -> None:
    st.markdown(_poster_html(movie), unsafe_allow_html=True)
    st.markdown(_movie_meta_html(movie), unsafe_allow_html=True)

    why = movie.get("why")
    if why:
        source = "LLM" if why.get("source") == "llm" else "rule-based"
        st.caption(f"🤖 *Why ({source}):* {why.get('text')}")

    mid = movie.get("movie_id")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("👍", key=f"{key}_like", help="Like"):
            _post_feedback("like", mid)
    with c2:
        if st.button("👎", key=f"{key}_dis", help="Dislike"):
            _post_feedback("dislike", mid)
    with c3:
        if st.button("➕", key=f"{key}_wl", help="Watchlist"):
            _post_feedback("watchlist", mid)
    with c4:
        if st.button("🧹", key=f"{key}_rm", help="Remove"):
            _post_feedback("remove", mid)
    st.write("")


def render_grid(movies: list[dict], key_prefix: str) -> None:
    if not movies:
        st.markdown(
            '<div class="cm-empty">No movies found. Try a different search.</div>',
            unsafe_allow_html=True,
        )
        return
    for row_start in range(0, len(movies), 3):
        cols = st.columns(3)
        for col, movie in zip(cols, movies[row_start : row_start + 3]):
            with col:
                key = f"{key_prefix}_{row_start}_{movie.get('movie_id')}"
                render_card(movie, key)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_for_you() -> None:
    st.markdown('<div class="cm-section">For you — hybrid recommendations</div>', unsafe_allow_html=True)
    col_btn, col_query = st.columns([1, 3])
    with col_btn:
        if st.button("✨ Get recommendations", use_container_width=True):
            with st.spinner("Running the hybrid pipeline ..."):
                if st.session_state.get("use_demo"):
                    st.session_state.recommendations = DEMO_RECOMMENDATIONS
                else:
                    st.session_state.recommendations = _get(
                        f"/api/recommend/{st.session_state.user_id}",
                        {"n": 9, "with_explanations": True},
                    )["recommendations"]
    with col_query:
        st.session_state.query = st.text_input(
            "Describe a mood or preference (optional)",
            value=st.session_state.query,
            placeholder='e.g. "a feel-good Hindi comedy about friendship"',
            label_visibility="collapsed",
        )
        if st.button("🎯 Recommend with preference", use_container_width=True):
            with st.spinner("Blending your preference into recommendations ..."):
                st.session_state.recommendations = _post(
                    "/api/recommend/query",
                    {
                        "user_id": st.session_state.user_id,
                        "query": st.session_state.query,
                        "n": 9,
                        "with_explanations": True,
                    },
                )["recommendations"]

    if not st.session_state.recommendations:
        st.markdown(
            '<div class="cm-empty">Hit "Get recommendations" to see your picks.</div>',
            unsafe_allow_html=True,
        )
    else:
        render_grid(st.session_state.recommendations, "rec")


def tab_indian() -> None:
    st.markdown(
        '<div class="cm-section">🇮🇳 Indian cinema — 379 films across 8 languages</div>',
        unsafe_allow_html=True,
    )
    col_lang, _ = st.columns([1, 3])
    with col_lang:
        language = st.selectbox("Language", INDIAN_LANGUAGES, index=0)

    params = {"origin": "indian", "n": 12}
    if language != "All":
        params["language"] = language
    with st.spinner("Loading trending Indian titles ..."):
        try:
            trending = _get("/api/movies/trending", params)["results"]
        except requests.RequestException as exc:
            trending = []
            st.error(f"Could not load trending movies: {exc}")

    if trending:
        st.markdown(f"**🔥 Trending in Indian cinema{f' · {language}' if language != 'All' else ''}**")
        render_grid(trending, "ind")

    st.markdown('<div class="cm-section">Search Indian cinema by mood</div>', unsafe_allow_html=True)
    mood = st.text_input(
        "What are you in the mood for?",
        placeholder='e.g. "a Malayalam crime thriller", "a Punjabi wedding comedy"',
    )
    if mood.strip() and st.button("🎬 Find Indian movies", use_container_width=True):
        with st.spinner("Searching Indian catalog ..."):
            try:
                results = _get(
                    "/api/movies/search",
                    {"q": mood, "origin": "indian", "n": 12},
                )["results"]
                render_grid(results, "mood")
            except requests.RequestException as exc:
                st.error(f"Search failed: {exc}")


def tab_search() -> None:
    st.markdown('<div class="cm-section">Semantic search — the whole catalog</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Search in natural language",
        placeholder="gritty 90s thriller with a mind-bending plot twist",
    )

    st.markdown('<div class="cm-section">Explore</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXPLORE_CARDS))
    for col, card in zip(cols, EXPLORE_CARDS):
        with col:
            if st.button(f"✨ {card['label']}", key=f"explore_{card['label']}", use_container_width=True):
                st.session_state.search_results = _get(
                    "/api/movies/search", {"q": card["query"], "n": 12}
                )["results"]
                st.session_state.search_query = card["query"]

    if query.strip() and st.button("🔍 Search", use_container_width=True):
        with st.spinner("Searching vectors ..."):
            try:
                st.session_state.search_results = _get(
                    "/api/movies/search", {"q": query, "n": 12}
                )["results"]
                st.session_state.search_query = query
            except requests.RequestException as exc:
                st.error(f"Search failed: {exc}")

    results = st.session_state.get("search_results")
    search_query = st.session_state.get("search_query")
    if search_query:
        st.markdown(f"**Results for: _{search_query}_**")
    if results:
        render_grid(results, "search")
    elif not query.strip():
        st.markdown(
            '<div class="cm-empty">Search or tap an explore card to find movies.</div>',
            unsafe_allow_html=True,
        )


def tab_profile() -> None:
    st.markdown('<div class="cm-section">Your profile</div>', unsafe_allow_html=True)
    if st.button("👤 Show my profile"):
        with st.spinner("Loading profile ..."):
            try:
                profile = _get(f"/api/user/{st.session_state.user_id}/profile")
            except requests.RequestException as exc:
                st.error(f"Profile failed: {exc}")
                return
        liked = profile.get("liked", [])
        watchlist = profile.get("watchlist", [])
        if liked:
            st.markdown(f"**Top liked ({len(liked)})**")
            for m in liked[:10]:
                stars = _stars(m.get("rating", 0.0) / 5.0)
                st.markdown(
                    f"- **{_escape(m.get('title', ''))}** &nbsp;{stars}&nbsp;"
                    f"<span class='cm-chip'>{_escape(m.get('year') and str(m.get('year')) or '')}</span>"
                    if m.get("year")
                    else f"- **{_escape(m.get('title', ''))}** &nbsp;{stars}",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="cm-empty">No movie history for this user yet.</div>', unsafe_allow_html=True)
        if watchlist:
            st.markdown(f"**Watchlist ({len(watchlist)})**")
            st.write(", ".join(str(w) for w in watchlist))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEMO_RECOMMENDATIONS = [
    {
        "movie_id": 1,
        "clean_title": "The Demo Odyssey",
        "year": 2022,
        "genres": ["Sci-Fi", "Adventure"],
        "score": 0.97,
        "why": {"source": "rule", "text": "Stunning visuals and an emotional core."},
    },
    {
        "movie_id": 2,
        "clean_title": "Hackers' Ball",
        "year": 2019,
        "genres": ["Thriller"],
        "score": 0.88,
        "why": {"source": "rule", "text": "Fast-paced plot, great for late-night viewing."},
    },
    {
        "movie_id": 3,
        "clean_title": "Monsoon of Ideas",
        "year": 2021,
        "genres": ["Drama", "Romance"],
        "score": 0.91,
        "why": {"source": "rule", "text": "Warm coming-of-age story with a strong soundtrack."},
    },
]

EXPLORE_CARDS = [
    {"label": "Mind-bending sci-fi", "query": "mind-bending science fiction with a plot twist"},
    {"label": "90s action thriller", "query": "high-octane 90s action thriller"},
    {"label": "Feel-good comedy", "query": "feel-good comedy about friendship"},
    {"label": "Bollywood romance", "query": "Hindi romantic drama with songs"},
    {"label": "Tamil crime saga", "query": "Tamil crime thriller gangster"},
    {"label": "Epic fantasy", "query": "epic fantasy adventure with a hero"},
]


def get_explore_cards() -> list[dict]:
    """One-click demo prompts shown on the Search tab (recruiter-friendly)."""
    return EXPLORE_CARDS


def main() -> None:
    st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="cm-hero"><h1>🎬 CineMatch AI</h1>'
        "<p>Hybrid recommendations · semantic search · Indian cinema · LLM explanations</p></div>",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("user_id", DEFAULT_USER)
    st.session_state.setdefault("recommendations", [])
    st.session_state.setdefault("query", "")

    with st.sidebar:
        st.markdown("### 🎛 Session")
        st.session_state.user_id = st.number_input(
            "User ID", min_value=1, value=st.session_state.user_id, step=1
        )
        try:
            health = _get("/health")
            st.success(
                f"API ok · **{health['movies']:,}** movies "
                f"({health.get('indian_movies', 0):,} Indian) · "
                f"vectors: {'✅' if health['vector_index'] else '❌'}"
            )
            if not health["vector_index"]:
                st.warning("Run `python scripts/index_vectors.py` to enable semantic search.")
        except requests.RequestException:
            st.error(f"Cannot reach API at {API_URL}. Start it with `python scripts/run_api.py`.")

        st.session_state.use_demo = st.checkbox("Use demo data (no backend)", value=False)
        st.markdown("---")
        st.markdown(
            "**About** — collaborative filtering (SVD + item-based CF), semantic "
            "embeddings over Qdrant, and rule-based explanations. MovieLens 100k "
            "+ a curated Indian catalog of 379 films."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 For You", "🇮🇳 Indian Cinema", "🔍 Search", "👤 Profile"]
    )
    with tab1:
        tab_for_you()
    with tab2:
        tab_indian()
    with tab3:
        tab_search()
    with tab4:
        tab_profile()


if __name__ == "__main__":
    main()
