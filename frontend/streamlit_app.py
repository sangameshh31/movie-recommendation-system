"""CineMatch AI — Streamlit frontend (IMDb-style dark theme).

Pages:

* **Home**     — hero + horizontal rails: Popular now, Top rated, New releases,
                 Indian cinema, Anime, TV series.
* **For You**  — hybrid recommendations + "why" explanations + feedback.
* **Indian**   — Indian cinema rails per language + mood search.
* **Anime**    — anime movies & series rails.
* **TV**       — TV series rails.
* **Search**   — natural-language semantic search + one-click explore cards.
* **Profile**  — the user's liked movies and watchlist.

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
    "Gujarati",
    "Urdu",
    "Assamese",
    "Oriya",
    "Bhojpuri",
]

# Deterministic gradient fallback palette (used when no poster exists).
PALETTE = [
    ("#37474f", "#121212"),
    ("#1e3a5f", "#0b1220"),
    ("#4a2c5f", "#170b20"),
    ("#5f2c3d", "#1d0b12"),
    ("#2c5f4a", "#0b1d15"),
    ("#5f4a2c", "#1d170b"),
    ("#3d2c5f", "#120b1d"),
    ("#2c3d5f", "#0b121d"),
]

IMDB_YELLOW = "#f5c518"

_CSS = """
<style>
:root {
  --bg: #0f1217;
  --panel: #1a1f26;
  --panel2: #151a20;
  --border: #2a313a;
  --text: #f2f2f2;
  --muted: #a0aab5;
  --accent: #f5c518;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(245,197,24,.07), transparent 60%),
    radial-gradient(900px 500px at 95% 0%, rgba(56,189,248,.05), transparent 55%),
    var(--bg);
}
[data-testid="stSidebar"] {
  background: var(--panel2);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 1.4rem; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: -.2px; }
a { color: var(--accent); }
::-webkit-scrollbar { width: 10px; height: 8px; }
::-webkit-scrollbar-track { background: var(--panel2); }
::-webkit-scrollbar-thumb { background: #3a4450; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ---- top logo band ---- */
.cm-logo { font-size: 1.7rem; font-weight: 800; letter-spacing: -1px; color: var(--accent); }
.cm-logo span { color: #fff; }
.cm-tagline { color: var(--muted); font-size: .85rem; margin-top: 2px; }

/* ---- nav pills ---- */
div[data-testid="stPills"] > div { gap: .4rem; }
div[data-testid="stPills"] button {
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--muted);
  border-radius: 999px;
  font-weight: 600;
  transition: all .15s ease;
}
div[data-testid="stPills"] button:hover {
  border-color: var(--accent);
  color: #fff;
}
div[data-testid="stPills"] button[aria-checked="true"] {
  background: var(--accent);
  border-color: var(--accent);
  color: #111;
}

/* ---- section headings (IMDb rail style) ---- */
.cm-rail-title {
  color: #fff; font-weight: 700; font-size: 1.25rem;
  margin: 1.1rem 0 .5rem 0; display: flex; align-items: baseline; gap: .5rem;
}
.cm-rail-title .cm-count { color: var(--muted); font-size: .8rem; font-weight: 400; }
.cm-rail-title .cm-link { color: var(--accent); font-size: .8rem; font-weight: 600; }

/* ---- horizontal scroller ---- */
.cm-rail { display: flex; gap: 12px; overflow-x: auto; padding: 2px 2px 12px 2px; }
.cm-tile {
  flex: 0 0 152px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.cm-tile:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 28px rgba(0,0,0,.55);
  border-color: var(--accent);
}
.cm-poster { position: relative; height: 228px; background: #0b0e12; }
.cm-poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.cm-poster .cm-fallback {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
}
.cm-poster .cm-fallback span {
  font-size: 4.6rem; font-weight: 900; color: rgba(255,255,255,.16);
}
.cm-badge {
  position: absolute; top: 6px; right: 6px;
  background: var(--accent); color: #111;
  font-weight: 800; font-size: .8rem; line-height: 1;
  padding: 3px 7px 3px 6px; border-radius: 5px;
  display: flex; align-items: center; gap: 3px;
  box-shadow: 0 1px 5px rgba(0,0,0,.6);
}
.cm-badge svg { width: 11px; height: 11px; }
.cm-type {
  position: absolute; bottom: 6px; left: 6px;
  background: rgba(0,0,0,.72); color: #fff;
  font-size: .62rem; font-weight: 700; letter-spacing: .6px;
  padding: 2px 6px; border-radius: 3px; text-transform: uppercase;
}
.cm-tile-body { padding: 8px 9px 10px 9px; }
.cm-tile-title {
  font-size: .85rem; font-weight: 600; color: #fff; line-height: 1.28;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden; min-height: 2.2em;
}
.cm-tile-sub { font-size: .76rem; color: var(--muted); margin-top: 4px; }

/* ---- hero ---- */
.cm-hero {
  display: flex; gap: 18px; align-items: stretch;
  background: linear-gradient(100deg, #1b222c 0%, #141a22 60%, #0f1217 100%);
  border: 1px solid var(--border); border-radius: 14px;
  padding: 16px; margin: .4rem 0 .2rem 0;
  box-shadow: inset 0 0 60px rgba(0,0,0,.35);
}
.cm-hero-poster { flex: 0 0 180px; }
.cm-hero-poster img { width: 100%; border-radius: 10px; box-shadow: 0 12px 30px rgba(0,0,0,.6); }
.cm-hero-body { padding: .4rem .6rem; }
.cm-hero-title { font-size: 2rem; font-weight: 800; color: #fff; line-height: 1.1; }
.cm-hero-sub { color: var(--muted); margin-top: .4rem; font-size: .9rem; }
.cm-hero-rating { margin-top: .8rem; display: flex; align-items: center; gap: .5rem; }
.cm-hero-big {
  background: var(--accent); color: #111; font-weight: 800; font-size: 1.2rem;
  padding: 4px 10px; border-radius: 6px; display: inline-flex; align-items: center; gap: 5px;
}
.cm-hero-genres { margin-top: .8rem; }

/* ---- chips ---- */
.cm-chip {
  display: inline-block;
  background: #232b35; border: 1px solid var(--border);
  color: #c8d1dc; border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0;
  font-size: .78rem; white-space: nowrap;
}
.cm-chip:hover { border-color: var(--accent); color: #fff; }
.cm-lang { background: #16233a; border: 1px solid #27466e; color: #8ec5ff; }
.cm-stars { color: var(--accent); letter-spacing: 1px; font-size: .85rem; }
.cm-score { color: var(--muted); font-size: .8rem; }
.cm-empty { text-align: center; color: var(--muted); padding: 3rem 0; font-size: 1.05rem; }

/* ---- inputs / buttons (IMDb form look) ---- */
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div, div[data-testid="stSlider"] [data-testid="stThumbValue"] {
  background: #20262e;
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 8px;
}
div[data-testid="stTextInput"] input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(245,197,24,.18); }
[data-testid="stExpander"] details { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stExpander"] summary { color: var(--text); }
.stButton > button {
  background: #1e242c; border: 1px solid var(--border); color: var(--text);
  border-radius: 8px; transition: all .15s ease; font-weight: 600;
}
.stButton > button:hover {
  background: var(--accent); border-color: var(--accent); color: #111;
  transform: translateY(-1px);
}

/* ---- search box ---- */
.cm-searchbox {
  margin: .6rem 0 .2rem 0;
}
.cm-searchbox input { padding: .8rem 1rem; font-size: 1.05rem; border-radius: 10px; }

/* ---- live-status footer ---- */
.cm-stats { color: var(--muted); font-size: .85rem; margin: 1rem 0; display: flex; gap: 1rem; flex-wrap: wrap; }
.cm-stats b { color: var(--accent); }
</style>
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "users.json")


def _load_users() -> dict[str, int]:
    try:
        import json

        with open(USERS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_users(users: dict[str, int]) -> None:
    import json

    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def _next_user_id() -> int:
    users = _load_users()
    used = set(users.values())
    candidate = 1000
    while candidate in used:
        candidate += 1
    return candidate


def _sign_in(username: str) -> bool:
    users = _load_users()
    uid = users.get(username.strip().lower())
    if uid is None:
        return False
    st.session_state.user = {"username": username.strip(), "user_id": int(uid)}
    st.session_state.user_id = int(uid)
    return True


def _sign_up(username: str) -> bool:
    name = username.strip()
    if not name:
        return False
    users = _load_users()
    key = name.lower()
    if key in users:
        st.session_state.user = {"username": name, "user_id": int(users[key])}
        st.session_state.user_id = int(users[key])
        return True
    uid = _next_user_id()
    users[key] = uid
    _save_users(users)
    st.session_state.user = {"username": name, "user_id": uid}
    st.session_state.user_id = uid
    return True


def _color_for(text: str) -> tuple[str, str]:
    digest = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return PALETTE[digest % len(PALETTE)]


def _escape(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _stars(score: float) -> str:
    n = max(0, min(5, round(score * 5)))
    return "★" * n + "☆" * (5 - n)


def _badge_html(vote: float | None) -> str:
    if vote is None:
        return ""
    return (
        f'<div class="cm-badge"><svg viewBox="0 0 32 32" fill="none">'
        f'<path d="M16 2l4 9 10 1-7 7 2 10-9-5-9 5 2-10-7-7 10-1z" fill="#111" stroke="#111"/>'
        f'</svg>{vote:.1f}</div>'
    )


def _type_tag(media_type: str | None) -> str:
    t = (media_type or "movie").strip()
    if t == "series":
        return '<span class="cm-type">Series</span>'
    return ""


def _poster_img_html(movie: dict, width: int = 342) -> tuple[str, str]:
    """Return (img_html, fallback_html) for a poster (real image or gradient).

    The fallback is hidden by default and only revealed via ``onerror`` so it
    never covers the poster image.
    """
    title = _escape(movie.get("clean_title") or movie.get("title") or "Untitled")
    c1, c2 = _color_for(title)
    initial = (movie.get("clean_title") or movie.get("title") or "?").strip()[:1].upper()
    poster = movie.get("poster_url")
    fallback = (
        f'<div class="cm-fallback" style="display:none;background:linear-gradient(160deg,{c1},{c2});">'
        f'<span>{_escape(initial)}</span></div>'
    )
    if poster:
        img = (
            f'<img src="{_escape(poster)}" alt="{title}" loading="lazy" '
            f'onerror="var f=this.nextElementSibling;if(f)f.style.display=\'flex\';this.remove();">'
        )
    else:
        img = ""
    return img, fallback


def _tile_html(movie: dict) -> str:
    title = _escape(movie.get("clean_title") or movie.get("title") or "Untitled")
    year = movie.get("year")
    lang = movie.get("language")
    img, fallback = _poster_img_html(movie)
    sub = f"{year}" if year else ""
    if lang:
        sub = f"{sub} · {_escape(lang)}" if sub else _escape(lang)
    return (
        f'<div class="cm-tile">'
        f'<div class="cm-poster">{img}{fallback}'
        f'{_badge_html(movie.get("vote_average"))}{_type_tag(movie.get("media_type"))}</div>'
        f'<div class="cm-tile-body">'
        f'<div class="cm-tile-title">{title}</div>'
        f'<div class="cm-tile-sub">{sub}</div>'
        f'</div></div>'
    )


def _rail(title: str, movies: list[dict], count: int | None = None) -> None:
    if not movies:
        return
    count_html = f'<span class="cm-count">{count:,}</span>' if count else ""
    st.markdown(
        f'<div class="cm-rail-title">{_escape(title)} {count_html}</div>',
        unsafe_allow_html=True,
    )
    tiles = "".join(_tile_html(m) for m in movies)
    st.markdown(f'<div class="cm-rail">{tiles}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTTP helpers (cached)
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict:
    resp = _http.get(f"{API_URL}{path}", params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = _http.post(f"{API_URL}{path}", json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_rail(
    sort_by: str,
    origin: str,
    media_type: str,
    language: str,
    n: int,
    genre: str = "",
) -> list[dict]:
    params = {"n": n, "sort_by": sort_by}
    if origin:
        params["origin"] = origin
    if media_type:
        params["media_type"] = media_type
    if language:
        params["language"] = language
    if genre:
        params["genre"] = genre
    try:
        return _get("/api/movies/trending", params)["results"]
    except requests.RequestException:
        return []


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_recommendations(user_id: int, n: int) -> list[dict]:
    try:
        return _get(
            f"/api/recommend/{user_id}", {"n": n, "with_explanations": True}
        )["recommendations"]
    except requests.RequestException:
        return []


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
# Interactive card (poster + meta + feedback buttons)
# ---------------------------------------------------------------------------

def render_card(movie: dict, key: str) -> None:
    title = _escape(movie.get("clean_title") or movie.get("title") or "Untitled")
    year = movie.get("year")
    img, fallback = _poster_img_html(movie)
    badge = _badge_html(movie.get("vote_average"))
    lang = movie.get("language")

    meta = []
    if lang:
        meta.append(f'<span class="cm-chip cm-lang">{_escape(lang)}</span>')
    for genre in movie.get("genres", [])[:4]:
        meta.append(f'<span class="cm-chip">{_escape(genre)}</span>')
    score = movie.get("score")
    if isinstance(score, (int, float)):
        meta.append(
            f'<span class="cm-stars">{_stars(score)}</span> '
            f'<span class="cm-score">match {score:.0%}</span>'
        )

    st.markdown(
        f'<div class="cm-tile" style="flex:0 0 100%; width:100%;">'
        f'<div class="cm-poster" style="height:300px;">{img}{fallback}{badge}'
        f'{_type_tag(movie.get("media_type"))}</div>'
        f'<div class="cm-tile-body">'
        f'<div class="cm-tile-title" style="min-height:0; -webkit-line-clamp:3; font-size:.95rem;">{title}'
        f'{" · " + str(year) if year else ""}</div>'
        f'<div style="line-height:2; margin-top:4px;">{" ".join(meta)}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

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
            '<div class="cm-empty">Nothing found. Try a different search.</div>',
            unsafe_allow_html=True,
        )
        return
    for row_start in range(0, len(movies), 3):
        cols = st.columns(3)
        for col, movie in zip(cols, movies[row_start : row_start + 3]):
            with col:
                key = f"{key_prefix}_{row_start}_{movie.get('movie_id')}"
                render_card(movie, key)


def render_grid_filtered(movies: list[dict], key_prefix: str) -> None:
    """Grid plus a client-side filter bar (genre / release year / min match)."""
    if not movies:
        render_grid(movies, key_prefix)
        return

    with st.expander("🎛 Filter results", expanded=False):
        c1, c2, c3 = st.columns(3)
        all_genres = sorted({g for m in movies for g in m.get("genres", []) if g})
        genre = c1.selectbox("Genre", ["All", *all_genres], key=f"{key_prefix}_f_genre")
        years = sorted({m.get("year") for m in movies if m.get("year")})
        if len(years) >= 2:
            lo, hi = int(years[0]), int(years[-1])
            yr = c2.slider("Release year", lo, hi, (lo, hi), key=f"{key_prefix}_f_year")
        else:
            yr = None
            c2.caption("—")
        min_score = c3.slider("Min match", 0.0, 1.0, 0.0, 0.05, key=f"{key_prefix}_f_score")

    filtered = [
        m
        for m in movies
        if (genre == "All" or genre in m.get("genres", []))
        and (yr is None or yr[0] <= (m.get("year") or 0) <= yr[1])
        and ((m.get("score") or 1.0) >= min_score)
    ]
    if not filtered:
        st.markdown(
            '<div class="cm-empty">Nothing matches those filters — widen the range.</div>',
            unsafe_allow_html=True,
        )
        return
    render_grid(filtered, key_prefix)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

def _hero_html(featured: dict | None) -> str:
    if not featured:
        return ""
    title = _escape(featured.get("clean_title") or featured.get("title") or "")
    year = featured.get("year")
    lang = featured.get("language")
    genres = ", ".join(featured.get("genres", [])[:3]) or "Featured"
    img, fallback = _poster_img_html(featured, width=500)
    poster = featured.get("poster_url")
    poster_html = img + fallback if poster else fallback
    vote = featured.get("vote_average")
    rating_html = ""
    if vote:
        rating_html = (
            f'<div class="cm-hero-rating"><div class="cm-hero-big">★ {vote:.1f}</div>'
            f'<span class="cm-hero-sub">{featured.get("media_type", "movie").title()}</span></div>'
        )
    sub = f"{year} · {_escape(lang)}" if lang else (f"{year}" if year else "")
    return (
        f'<div class="cm-hero">'
        f'<div class="cm-hero-poster">{poster_html}</div>'
        f'<div class="cm-hero-body">'
        f'<div class="cm-hero-sub">FEATURED</div>'
        f'<div class="cm-hero-title">{title}</div>'
        f'<div class="cm-hero-sub">{sub}</div>'
        f'{rating_html}'
        f'<div class="cm-hero-genres"><span class="cm-chip">{_escape(genres)}</span></div>'
        f'</div></div>'
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home() -> None:
    popular = _fetch_rail("popularity", "", "movie", "", 20)
    if popular:
        st.markdown(_hero_html(popular[0]), unsafe_allow_html=True)

    _rail("🔥 Popular now", popular)
    _rail("⭐ Top rated", _fetch_rail("rating", "", "movie", "", 20))
    _rail("🆕 New releases", _fetch_rail("new", "", "", "", 20))
    _rail("🇮🇳 Indian cinema", _fetch_rail("popularity", "indian", "", "", 20), count=st.session_state.health.get("indian_movies"))
    _rail("⭐ Anime", _fetch_rail("popularity", "anime", "", "", 20), count=st.session_state.health.get("anime"))
    _rail("📺 TV series", _fetch_rail("popularity", "series", "", "", 20), count=st.session_state.health.get("series"))

    st.markdown(
        '<div class="cm-stats">'
        f'<span><b>{st.session_state.health.get("movies", 0):,}</b> titles</span>'
        f'<span><b>{st.session_state.health.get("indian_movies", 0):,}</b> Indian</span>'
        f'<span><b>{st.session_state.health.get("anime", 0):,}</b> anime</span>'
        f'<span><b>{st.session_state.health.get("series", 0):,}</b> series</span>'
        '<span>Search anything · enjoy everything 🎬</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def page_for_you() -> None:
    st.markdown('<div class="cm-rail-title">🎯 For you — hybrid recommendations</div>', unsafe_allow_html=True)
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
        render_grid_filtered(st.session_state.recommendations, "rec")


def page_indian() -> None:
    h = st.session_state.get("health") or {}
    n_indian = h.get("indian_movies")
    header = (
        f"🇮🇳 Indian cinema — {n_indian:,} films"
        if n_indian
        else "🇮🇳 Indian cinema"
    )
    st.markdown(f'<div class="cm-rail-title">{header}</div>', unsafe_allow_html=True)
    col_lang, _ = st.columns([1, 3])
    with col_lang:
        language = st.selectbox("Language", INDIAN_LANGUAGES, index=0)

    lang = "" if language == "All" else language
    _rail("🔥 Trending", _fetch_rail("popularity", "indian", "", lang, 20))
    _rail("⭐ Top rated", _fetch_rail("rating", "indian", "", lang, 16))
    _rail("🆕 Recent", _fetch_rail("new", "indian", "", lang, 16))

    st.markdown('<div class="cm-rail-title">Search Indian cinema by mood</div>', unsafe_allow_html=True)
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
                render_grid_filtered(results, "mood")
            except requests.RequestException as exc:
                st.error(f"Search failed: {exc}")


def page_anime() -> None:
    st.markdown('<div class="cm-rail-title">⭐ Anime & Japanese animation</div>', unsafe_allow_html=True)
    _rail("🔥 Popular", _fetch_rail("popularity", "anime", "", "", 20))
    _rail("⭐ Top rated", _fetch_rail("rating", "anime", "", "", 16))
    _rail("📺 Anime series", _fetch_rail("popularity", "anime", "series", "", 16))
    _rail("🆕 New", _fetch_rail("new", "anime", "", "", 16))


GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "Music",
]


def page_genres() -> None:
    st.markdown('<div class="cm-rail-title">🎭 Browse by genre</div>', unsafe_allow_html=True)
    selected = st.pills(
        "Genre",
        GENRES,
        selection_mode="single",
        default=None,
        key="genre_pick",
        label_visibility="collapsed",
    )
    genre = selected or ""
    if not genre:
        st.markdown(
            '<div class="cm-empty">Pick a genre above to explore the best titles in it.</div>',
            unsafe_allow_html=True,
        )
        return

    _rail(f"🔥 Popular {genre}", _fetch_rail("popularity", "", "", "", 18, genre))
    _rail(f"⭐ Top rated {genre}", _fetch_rail("rating", "", "", "", 18, genre))
    _rail(f"🇮🇳 {genre} in Indian cinema", _fetch_rail("popularity", "indian", "", "", 16, genre))
    _rail(f"🆕 Recent {genre}", _fetch_rail("new", "", "", "", 16, genre))

    st.markdown(f'<div class="cm-rail-title">Explore {genre} — full grid</div>', unsafe_allow_html=True)
    with st.spinner(f"Loading {genre} ..."):
        grid = _fetch_rail("rating", "", "", "", 24, genre)
    render_grid_filtered(grid, "genre")


def page_tv() -> None:
    st.markdown('<div class="cm-rail-title">📺 TV series</div>', unsafe_allow_html=True)
    _rail("🔥 Popular", _fetch_rail("popularity", "series", "", "", 20))
    _rail("⭐ Top rated", _fetch_rail("rating", "series", "", "", 20))
    _rail("🆕 New", _fetch_rail("new", "series", "", "", 16))


def page_search() -> None:
    st.markdown('<div class="cm-rail-title">🔍 Search the whole catalog</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Search in natural language",
        value=st.session_state.get("search_query", ""),
        key="search_box",
        placeholder="gritty 90s thriller · anime adventure · Tamil crime saga · feel-good sitcom",
        label_visibility="collapsed",
    )

    st.markdown('<div class="cm-rail-title">✨ Explore</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXPLORE_CARDS))
    for col, card in zip(cols, EXPLORE_CARDS):
        with col:
            if st.button(f"{card['label']}", key=f"explore_{card['label']}", use_container_width=True):
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
        render_grid_filtered(results, "search")
    elif not query.strip():
        st.markdown(
            '<div class="cm-empty">Search or tap an explore card to find movies.</div>',
            unsafe_allow_html=True,
        )


def page_profile() -> None:
    current = st.session_state.get("user", {})
    name = _escape(current.get("username", "guest"))
    uid = current.get("user_id", DEFAULT_USER)
    st.markdown(
        f'<div class="cm-rail-title">👤 {name} · user {uid}</div>',
        unsafe_allow_html=True,
    )
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
                line = f"- **{_escape(m.get('title', ''))}** &nbsp;{stars}"
                if m.get("year"):
                    line += f"&nbsp;<span class='cm-chip'>{m.get('year')}</span>"
                st.markdown(line, unsafe_allow_html=True)
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
        "vote_average": 8.2,
        "why": {"source": "rule", "text": "Stunning visuals and an emotional core."},
    },
    {
        "movie_id": 2,
        "clean_title": "Hackers' Ball",
        "year": 2019,
        "genres": ["Thriller"],
        "score": 0.88,
        "vote_average": 7.4,
        "why": {"source": "rule", "text": "Fast-paced plot, great for late-night viewing."},
    },
    {
        "movie_id": 3,
        "clean_title": "Monsoon of Ideas",
        "year": 2021,
        "genres": ["Drama", "Romance"],
        "score": 0.91,
        "vote_average": 7.9,
        "why": {"source": "rule", "text": "Warm coming-of-age story with a strong soundtrack."},
    },
]

EXPLORE_CARDS = [
    {"label": "🎬 Mind-bending sci-fi", "query": "mind-bending science fiction with a plot twist"},
    {"label": "🔥 90s action", "query": "high-octane 90s action thriller"},
    {"label": "😂 Feel-good comedy", "query": "feel-good comedy about friendship"},
    {"label": "💞 Bollywood romance", "query": "Hindi romantic drama with songs"},
    {"label": "🔪 Tamil crime", "query": "Tamil crime thriller gangster"},
    {"label": "🐉 Anime adventure", "query": "anime adventure with a hero"},
    {"label": "📺 Binge-worthy TV", "query": "binge-worthy crime drama tv series"},
    {"label": "🎨 Animation", "query": "animated family cartoon movie"},
]


def main() -> None:
    st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="cm-logo">CineMatch<span>.ai</span></div>'
        '<div class="cm-tagline">Recommendations, search & ratings for movies, anime, series and Indian cinema</div>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("user_id", DEFAULT_USER)
    st.session_state.setdefault("user", {"username": "guest", "user_id": DEFAULT_USER})
    st.session_state.setdefault("recommendations", [])
    st.session_state.setdefault("query", "")
    st.session_state.setdefault("page", "Home")

    with st.sidebar:
        st.markdown("### 🔐 Account")
        with st.expander("Sign in / Sign up", expanded=True):
            username = st.text_input(
                "Username",
                key="auth_username",
                placeholder="your name",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Sign in", use_container_width=True, key="btn_signin"):
                    if not _sign_in(username):
                        st.error("No account with that name — tap Sign up instead.")
            with c2:
                if st.button("Sign up", use_container_width=True, key="btn_signup"):
                    if not _sign_up(username):
                        st.error("Enter a username first.")
        current = st.session_state.get("user", {})
        st.markdown(
            f"Signed in as **{_escape(current.get('username', 'guest'))}** "
            f"(user **{current.get('user_id', DEFAULT_USER)}**)",
            unsafe_allow_html=True,
        )
        if st.button("Sign out", key="btn_signout"):
            st.session_state.user = {"username": "guest", "user_id": DEFAULT_USER}
            st.session_state.user_id = DEFAULT_USER
            st.rerun()

        with st.expander("🎛 Session (demo)", expanded=False):
            st.session_state.user_id = st.number_input(
                "User ID", min_value=1, value=st.session_state.user_id, step=1
            )
            st.session_state.use_demo = st.checkbox("Use demo data (no backend)", value=False)

        try:
            health = _get("/health")
            st.session_state.health = health
            st.success(
                f"API ok · **{health['movies']:,}** titles "
                f"({health.get('indian_movies', 0):,} Indian · "
                f"{health.get('anime', 0):,} anime · {health.get('series', 0):,} series) · "
                f"vectors: {'✅' if health['vector_index'] else '❌'}"
            )
            if not health["vector_index"]:
                st.warning("Run `python scripts/index_vectors.py` to enable semantic search.")
        except requests.RequestException:
            st.session_state.health = {}
            st.error(f"Cannot reach API at {API_URL}. Start it with `python scripts/run_api.py`.")

        st.markdown("---")
        h = st.session_state.get("health") or {}
        n_movies = h.get("movies")
        if n_movies:
            st.markdown(
                f"**About** — hybrid CF (SVD + item-based), semantic search over Qdrant, "
                f"rule-based explanations. A TMDB-expanded catalog of **{n_movies:,} titles** "
                f"(**{h.get('indian_movies', 0):,} Indian** across 13 languages + Hollywood, "
                f"anime, animation and TV series)."
            )
        else:
            st.markdown("**About** — MovieLens + a TMDB-expanded catalog.")

    # Global search bar (visible on every page, IMDb-style header search).
    sc1, sc2 = st.columns([5, 1])
    with sc1:
        global_query = st.text_input(
            "Search",
            value=st.session_state.get("search_query", ""),
            key="global_query",
            placeholder='Search anything — e.g. "a Tamil crime thriller", "anime adventure", "Breaking Bad"',
            label_visibility="collapsed",
        )
    with sc2:
        do_search = st.button("🔍 Search", use_container_width=True, key="btn_global_search")
    if do_search and global_query.strip():
        with st.spinner("Searching the catalog ..."):
            try:
                st.session_state.search_results = _get(
                    "/api/movies/search", {"q": global_query.strip(), "n": 12}
                )["results"]
                st.session_state.search_query = global_query.strip()
                st.session_state["nav"] = "Search"
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Search failed: {exc}")

    page = st.pills(
        "Navigate",
        ["Home", "For You", "Indian", "Anime", "TV", "Genres", "Search", "Profile"],
        selection_mode="single",
        default="Home",
        key="nav",
        label_visibility="collapsed",
    )
    st.session_state.page = page or st.session_state.page

    if page == "Home":
        page_home()
    elif page == "For You":
        page_for_you()
    elif page == "Indian":
        page_indian()
    elif page == "Anime":
        page_anime()
    elif page == "TV":
        page_tv()
    elif page == "Genres":
        page_genres()
    elif page == "Search":
        page_search()
    elif page == "Profile":
        page_profile()


if __name__ == "__main__":
    main()
