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
from urllib.parse import quote

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

/* ---- clickable tiles (whole card links to the detail page) ---- */
a.cm-tile-link { display: block; text-decoration: none; color: inherit; }
a.cm-tile-link:hover .cm-tile {
  transform: translateY(-5px);
  box-shadow: 0 12px 28px rgba(0,0,0,.55);
  border-color: var(--accent);
}
.cm-tile-link .cm-tile:hover { border-color: var(--accent); }

/* ---- movie detail page ---- */
.cm-detail-hero {
  display: flex; gap: 22px; align-items: stretch;
  background: linear-gradient(110deg, #1c2430 0%, #131920 55%, #0f1217 100%);
  border: 1px solid var(--border); border-radius: 16px;
  padding: 20px; margin: .5rem 0 1rem 0;
  position: relative; overflow: hidden;
}
.cm-detail-hero .cm-bg {
  position: absolute; inset: 0; background-size: cover; background-position: 60% 18%;
  opacity: .25; filter: blur(2px);
}
.cm-detail-hero .cm-shade {
  position: absolute; inset: 0;
  background: linear-gradient(115deg, rgba(12,15,20,.97) 28%, rgba(12,15,20,.55) 62%, rgba(12,15,20,.85));
}
.cm-detail-poster { flex: 0 0 250px; position: relative; z-index: 1; }
.cm-detail-poster img { width: 100%; border-radius: 12px; box-shadow: 0 16px 40px rgba(0,0,0,.65); }
.cm-detail-body { position: relative; z-index: 1; padding: .2rem .4rem; min-width: 0; }
.cm-detail-title { font-size: 2.3rem; font-weight: 800; color: #fff; line-height: 1.08; letter-spacing: -.5px; }
.cm-detail-meta {
  color: var(--muted); margin-top: .5rem; font-size: .95rem;
  display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
}
.cm-detail-meta b { color: var(--accent); }
.cm-tagline { color: #c6cfd9; font-style: italic; font-size: 1.05rem; margin-top: .7rem; }
.cm-overview { color: #e6e9ee; font-size: 1rem; line-height: 1.6; margin-top: .6rem; max-width: 46rem; }
.cm-crew { margin-top: 1rem; font-size: .92rem; color: var(--muted); }
.cm-crew b { color: #fff; font-weight: 700; }
.cm-crew .cm-chip { margin: 2px 4px 2px 0; }
.cm-detail-rating { display: inline-flex; align-items: center; gap: 6px; }
.cm-detail-rating .cm-hero-big { font-size: 1.15rem; }
.cm-note { color: var(--muted); font-size: .85rem; margin-top: .4rem; }

/* ---- cast strip ---- */
.cm-cast { display: flex; gap: 12px; overflow-x: auto; padding: 2px 2px 12px 2px; }
.cm-cast-card { flex: 0 0 112px; text-align: center; }
.cm-cast-card img { width: 100%; height: 152px; object-fit: cover; border-radius: 8px; background: #0b0e12; }
.cm-cast-name { font-size: .8rem; font-weight: 600; color: #fff; margin-top: 5px; line-height: 1.2; }
.cm-cast-char { font-size: .72rem; color: var(--muted); margin-top: 2px; line-height: 1.2; }

/* ---- auth card ---- */
.cm-auth {
  max-width: 440px; margin: 2.4rem auto;
  background: linear-gradient(180deg, #1a212b, #141a21);
  border: 1px solid var(--border); border-radius: 16px;
  padding: 1.9rem 1.7rem; text-align: center;
}
.cm-auth h3 { margin-bottom: .3rem; color: #fff; }
.cm-auth p { color: var(--muted); font-size: .9rem; margin-bottom: 1.2rem; }
.cm-auth .cm-auth-who { color: var(--muted); font-size: .95rem; margin: .9rem 0 1.1rem 0; }

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

_DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
USERS_FILE = os.path.join(_DATA_DIR, "users.json")


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
    st.session_state["onboarding"] = True
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


def _chips(names: list[str]) -> str:
    return "".join(f'<span class="cm-chip">{_escape(x)}</span>' for x in names)


def _person_link(name: str) -> str:
    """A clickable director/cast chip that deep-links to the person page."""
    if not name:
        return ""
    href = f"?person={quote(str(name))}"
    return (
        f'<a class="cm-chip" href="{href}" '
        f'title="Explore {_escape(name)} on TMDB">{_escape(name)}</a>'
    )


def _person_chips(names: list[str]) -> str:
    return "".join(_person_link(n) for n in names)


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
    mid = movie.get("movie_id")
    title = _escape(movie.get("clean_title") or movie.get("title") or "Untitled")
    year = movie.get("year")
    lang = movie.get("language")
    img, fallback = _poster_img_html(movie)
    sub = f"{year}" if year else ""
    if lang:
        sub = f"{sub} · {_escape(lang)}" if sub else _escape(lang)
    tile = (
        f'<div class="cm-tile">'
        f'<div class="cm-poster">{img}{fallback}'
        f'{_badge_html(movie.get("vote_average"))}{_type_tag(movie.get("media_type"))}</div>'
        f'<div class="cm-tile-body">'
        f'<div class="cm-tile-title">{title}</div>'
        f'<div class="cm-tile-sub">{sub}</div>'
        f'</div></div>'
    )
    if mid:
        return f'<a class="cm-tile-link" href="?detail={mid}" title="View details">{tile}</a>'
    return tile


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
# Backend helpers (in-process on Streamlit Cloud; cached so the engine is
# loaded once per app, not once per rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _backend() -> "Backend":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from local_backend import Backend

    return Backend()


def _get(path: str, params: dict | None = None) -> dict:
    return _backend().get(path, params)


def _post(path: str, body: dict) -> dict:
    return _backend().post(path, body)


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


def _post_feedback(action: str, movie_id: int, value: float | None = None) -> None:
    try:
        body = {
            "user_id": st.session_state.user_id,
            "movie_id": movie_id,
            "action": action,
        }
        if value is not None:
            body["value"] = float(value)
        _post("/api/feedback", body)
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

    mid = movie.get("movie_id")
    poster_block = (
        f'<a class="cm-tile-link" href="?detail={mid}" title="View details">'
        f'<div class="cm-poster" style="height:300px;">{img}{fallback}{badge}'
        f'{_type_tag(movie.get("media_type"))}</div></a>'
    )
    st.markdown(
        f'<div class="cm-tile" style="flex:0 0 100%; width:100%;">'
        f'{poster_block}'
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

    because_of = movie.get("because_of")
    if because_of:
        titles = [b.get("title") for b in because_of if b.get("title")]
        if titles:
            st.caption(f"🎯 Because you liked: {', '.join(titles[:2])}")

    if mid:
        if st.button("🎬 Details", key=f"{key}_det", use_container_width=True):
            st.session_state.detail = int(mid)
            st.rerun()

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
        f'<div class="cm-hero-genres"><span class="cm-chip">{_escape(genres)}</span>'
        f'<a href="?detail={featured.get("movie_id")}" class="cm-chip">More info →</a></div>'
        f'</div></div>'
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _surprise_block() -> None:
    """A one-click personalized pick with a re-roll button."""
    st.markdown('<div class="cm-rail-title">🎲 Surprise me</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        if st.button("🎲 Give me a surprise pick", use_container_width=True):
            with st.spinner("Rolling the dice ..."):
                try:
                    st.session_state.surprise = _get(
                        "/api/surprise", {"user_id": st.session_state.user_id}
                    )["pick"]
                except requests.RequestException as exc:
                    st.error(f"Surprise failed: {exc}")
    with sc2:
        st.markdown(
            '<div class="cm-note" style="margin-top:8px;">One pick, tuned to your taste — '
            'roll again for a fresh option.</div>',
            unsafe_allow_html=True,
        )
    pick = st.session_state.get("surprise")
    if pick:
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown(
                f'<a class="cm-tile-link" href="?detail={pick.get("movie_id")}" title="View details">'
                f'{_tile_html(pick)}</a>',
                unsafe_allow_html=True,
            )
        with col_b:
            why = pick.get("why")
            if why:
                st.caption(f"🤖 {why.get('text')}")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👍 Like", key="surprise_like", use_container_width=True):
                    _post_feedback("like", pick.get("movie_id"))
            with c2:
                if st.button("➕ Watchlist", key="surprise_wl", use_container_width=True):
                    _post_feedback("watchlist", pick.get("movie_id"))
            with c3:
                if st.button("🎲 Roll again", key="surprise_again", use_container_width=True):
                    with st.spinner("Rolling the dice ..."):
                        st.session_state.surprise = _get(
                            "/api/surprise", {"user_id": st.session_state.user_id}
                        )["pick"]
                    st.rerun()
        st.write("")


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

    _surprise_block()

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


def _cast_html(c: dict) -> str:
    name = c.get("name") or "?"
    character = c.get("character") or ""
    c1, c2 = _color_for(name)
    profile = c.get("profile_url")
    img = ""
    if profile:
        img = (
            f'<img src="{_escape(profile)}" alt="{_escape(name)}" loading="lazy" '
            f'style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;'
            f'border-radius:8px;" onerror="this.remove();">'
        )
    card = (
        f'<div class="cm-cast-card">'
        f'<div style="height:152px;position:relative;border-radius:8px;'
        f'background:linear-gradient(160deg,{c1},{c2});'
        f'display:flex;align-items:center;justify-content:center;">'
        f'<span style="font-size:2.4rem;font-weight:800;color:rgba(255,255,255,.2)">'
        f'{_escape(name[:1].upper())}</span>{img}</div>'
        f'<div class="cm-cast-name">{_escape(name)}</div>'
        f'<div class="cm-cast-char">{_escape(character)}</div>'
        f'</div>'
    )
    if name != "?":
        return (
            f'<a href="?person={quote(str(name))}" title="Explore {_escape(name)}" '
            f'style="text-decoration:none;color:inherit;">{card}</a>'
        )
    return card


def page_movie_detail(movie_id: int) -> None:
    st.markdown('<div class="cm-rail-title" style="margin-top:0;">🎬 Movie details</div>', unsafe_allow_html=True)
    if st.button("← Back to browsing", key="btn_detail_back"):
        st.session_state.pop("detail", None)
        st.query_params.clear()
        st.rerun()
    try:
        data = _get(f"/api/movies/{movie_id}/details")
    except requests.RequestException as exc:
        st.error(f"Could not load details: {exc}")
        return

    m = data.get("movie") or {}
    d = data.get("details") or {}
    similar = data.get("similar") or []

    title = _escape(m.get("clean_title") or m.get("title") or "Untitled")
    year = m.get("year")
    media_type = m.get("media_type", "movie")
    vote = m.get("vote_average")
    lang = m.get("language")
    overview = d.get("overview") or ""
    tagline = d.get("tagline") or ""
    runtime = d.get("runtime_min")
    release_date = d.get("release_date") or ""
    status = d.get("status") or ""
    backdrop = d.get("backdrop_url")
    genres = list(dict.fromkeys(list(d.get("genres") or []) + list(m.get("genres") or [])))
    director = d.get("director") or []
    producers = d.get("producers") or []
    writers = d.get("writers") or []
    cast = d.get("cast") or []
    seasons = d.get("seasons")
    episodes = d.get("episodes")

    img, fallback = _poster_img_html(m, width=500)

    meta_bits = []
    if year:
        meta_bits.append(str(year))
    if runtime:
        meta_bits.append(f"{int(runtime)} min")
    if media_type == "series":
        extra = []
        if seasons:
            extra.append(f"{seasons} season{'s' if seasons != 1 else ''}")
        if episodes:
            extra.append(f"{episodes} episodes")
        meta_bits.append("Series" + (f" · {', '.join(extra)}" if extra else ""))
    if release_date:
        meta_bits.append(release_date)
    if status:
        meta_bits.append(status)
    if lang:
        meta_bits.append(_escape(lang))

    rating_html = ""
    if vote:
        rating_html = (
            f'<div class="cm-detail-rating" style="margin-top:.8rem;">'
            f'<div class="cm-hero-big">★ {vote:.1f}</div>'
            f'<span class="cm-note">TMDB rating</span></div>'
        )

    crew_html = ""
    if director:
        crew_html += f'<div class="cm-crew"><b>Director</b>: {_person_chips(director)}</div>'
    if producers:
        crew_html += f'<div class="cm-crew"><b>Producer</b>: {_person_chips(producers[:5])}</div>'
    if writers:
        crew_html += f'<div class="cm-crew"><b>Writer</b>: {_person_chips(writers[:5])}</div>'
    genre_html = _chips(genres) if genres else ""

    bg_style = f'background-image:url("{_escape(backdrop)}");' if backdrop else ""
    bg_div = f'<div class="cm-bg" style="{bg_style}"></div>' if backdrop else ""
    overview_html = overview if overview else "No plot synopsis available for this title yet."
    tagline_html = f'<div class="cm-tagline">“{_escape(tagline)}”</div>' if tagline else ""
    crew_genre_html = (
        f'<div class="cm-crew" style="margin-top:1rem;">{genre_html}</div>' if genre_html else ""
    )
    st.markdown(
        f'<div class="cm-detail-hero">'
        f'{bg_div}<div class="cm-shade"></div>'
        f'<div class="cm-detail-poster">{img}{fallback}</div>'
        f'<div class="cm-detail-body">'
        f'<div class="cm-detail-title">{title}</div>'
        f'<div class="cm-detail-meta">{" · ".join(meta_bits)}</div>'
        f'{rating_html}'
        f'{tagline_html}'
        f'<div class="cm-overview">{_escape(overview_html)}</div>'
        f'{crew_genre_html}'
        f'{crew_html}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    watched = movie_id in st.session_state.get("watched_set", set())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("👍 Like", key="dt_like", use_container_width=True):
            _post_feedback("like", movie_id)
    with c2:
        if st.button("➕ Watchlist", key="dt_wl", use_container_width=True):
            _post_feedback("watchlist", movie_id)
    with c3:
        if st.button("👎 Not for me", key="dt_dis", use_container_width=True):
            _post_feedback("dislike", movie_id)
    with c4:
        if st.button(
            "✓ Watched" if watched else "🫥 Mark watched",
            key="dt_watched",
            use_container_width=True,
        ):
            _post_feedback("unwatched" if watched else "watched", movie_id)

    st.write("")
    rc1, rc2 = st.columns([2, 3])
    with rc1:
        star_val = st.select_slider(
            "⭐ Your rating",
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=st.session_state.get(f"star_{movie_id}", 3.5),
            key=f"star_slider_{movie_id}",
            label_visibility="collapsed",
        )
        st.session_state[f"star_{movie_id}"] = star_val
    with rc2:
        st.markdown(
            '<div class="cm-note" style="margin-top:26px;">Slide to your rating, then:'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("💾 Save rating", key=f"dt_rate_{movie_id}", use_container_width=True):
            _post_feedback("rate", movie_id, value=star_val)

    trailer_url = d.get("trailer_url")
    homepage = d.get("homepage")
    imdb_id = d.get("imdb_id")
    tmdb_id = d.get("tmdb_id")
    links = []
    if trailer_url:
        links.append(("▶ Trailer", trailer_url))
    if homepage:
        links.append(("🏠 Official site", homepage))
    if imdb_id:
        links.append(("🎥 IMDb", f"https://www.imdb.com/title/{imdb_id}/"))
    if tmdb_id:
        kind = "tv" if media_type == "series" else "movie"
        links.append(("🎬 TMDB", f"https://www.themoviedb.org/{kind}/{tmdb_id}"))
    if links:
        st.markdown('<div class="cm-rail-title">🔗 Watch & explore</div>', unsafe_allow_html=True)
        for l1, l2 in links:
            st.link_button(l1, l2, use_container_width=True)

    if cast:
        st.markdown('<div class="cm-rail-title">🎭 Cast</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="cm-cast">{"".join(_cast_html(c) for c in cast)}</div>',
            unsafe_allow_html=True,
        )

    if similar:
        _rail("🎯 More like this", similar)


def page_person(name: str) -> None:
    st.markdown('<div class="cm-rail-title" style="margin-top:0;">🎬 People</div>', unsafe_allow_html=True)
    back = st.session_state.get("person_back")
    if back:
        if st.button("← Back to movie", key="btn_person_back"):
            st.session_state.pop("person", None)
            st.session_state["detail"] = int(back)
            st.query_params.clear()
            st.rerun()
    else:
        if st.button("← Back to browsing", key="btn_person_back2"):
            st.session_state.pop("person", None)
            st.query_params.clear()
            st.rerun()

    with st.spinner(f"Looking up {name} ..."):
        try:
            person = _get("/api/people/search", {"q": name})
        except requests.RequestException as exc:
            st.error(f"Could not find that person: {exc}")
            return

    pname = person.get("name") or name
    profile_url = person.get("profile_url")
    department = person.get("department") or ""
    biography = person.get("biography") or ""
    catalog = person.get("catalog_movies") or []
    known_works = person.get("known_works") or []

    c1, c2 = st.columns([1, 3])
    with c1:
        if profile_url:
            st.image(profile_url, width=260)
        else:
            c1a, c1b = _color_for(pname)
            st.markdown(
                f'<div style="height:260px;border-radius:14px;'
                f'background:linear-gradient(160deg,{c1a},{c1b});'
                f'display:flex;align-items:center;justify-content:center;">'
                f'<span style="font-size:6rem;font-weight:800;color:rgba(255,255,255,.25)">'
                f'{_escape(pname[:1].upper())}</span></div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(
            f'<div class="cm-detail-title">{_escape(pname)}</div>',
            unsafe_allow_html=True,
        )
        if department:
            st.markdown(
                f'<div class="cm-detail-meta"><b>{_escape(department)}</b></div>',
                unsafe_allow_html=True,
            )
        bio = biography if biography else "No biography available for this person."
        st.markdown(f'<div class="cm-overview">{_escape(bio)}</div>', unsafe_allow_html=True)

    if catalog:
        _rail("🎥 In your catalog", catalog, count=len(catalog))
        st.markdown('<div class="cm-rail-title">Explore — full grid</div>', unsafe_allow_html=True)
        render_grid(catalog, "person")

    if known_works and len(known_works) > len(catalog):
        st.markdown('<div class="cm-rail-title">✨ More from this person</div>', unsafe_allow_html=True)
        tmdb_to_local = {c.get("tmdb_id"): c.get("movie_id") for c in catalog if c.get("tmdb_id")}
        tiles = []
        for k in known_works[:12]:
            mid = tmdb_to_local.get(k.get("tmdb_id"))
            tiles.append(_tile_html({**k, "movie_id": mid} if mid else k))
        if tiles:
            st.markdown(f'<div class="cm-rail">{"".join(tiles)}</div>', unsafe_allow_html=True)


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
        with st.expander("💬 Refine this search (conversational)", expanded=False):
            st.caption(
                "Tell the engine what to add or remove — e.g. add “more like Christopher "
                "Nolan movies” or remove “no romance”. It re-searches with those "
                "concepts mixed in."
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                add_text = st.text_input("➕ Add more like ...", key="refine_add")
            with rc2:
                rem_text = st.text_input("➖ Less of ...", key="refine_rem")
            if st.button("🔁 Refine results", key="btn_refine", use_container_width=True):
                additions = [t for t in add_text.split(",") if t.strip()]
                removals = [t for t in rem_text.split(",") if t.strip()]
                with st.spinner("Nudging the search vector ..."):
                    try:
                        st.session_state.search_results = _post(
                            "/api/search/refine",
                            {
                                "seed": search_query,
                                "additions": additions,
                                "removals": removals,
                                "n": 12,
                            },
                        )["results"]
                        st.session_state.refined = True
                    except requests.RequestException as exc:
                        st.error(f"Refine failed: {exc}")
    elif not query.strip():
        st.markdown(
            '<div class="cm-empty">Search or tap an explore card to find movies.</div>',
            unsafe_allow_html=True,
        )


def _auth_card() -> None:
    """A centered sign-in / sign-up card for the Profile page."""
    st.markdown(
        '<div class="cm-auth">'
        '<h3>🔐 Welcome to CineMatch</h3>'
        '<p>Sign in to save your likes, build a watchlist and get picks that learn from you.</p>'
        '<p class="cm-auth-who">You are browsing as <b>guest</b>.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    username = st.text_input(
        "Username",
        key="auth_username",
        placeholder="your name, e.g. aisha",
        label_visibility="collapsed",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Sign in", use_container_width=True, key="btn_signin"):
            if not _sign_in(username):
                st.error("No account with that name — tap “Sign up” instead.")
            else:
                st.rerun()
    with c2:
        if st.button("Sign up", use_container_width=True, key="btn_signup"):
            if not _sign_up(username):
                st.error("Enter a username first.")
            else:
                st.rerun()
    st.caption("Just type a name and “Sign up” to create an account — no password needed.")


def _onboarding_picker() -> None:
    """First-sign-in flow: pick favorite movies to seed the taste profile."""
    st.markdown(
        '<div class="cm-auth">'
        '<h3>🎉 Welcome aboard!</h3>'
        '<p>Pick a few movies you already love so CineMatch can tune your picks '
        'from day one. You can change these anytime.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    candidates = _fetch_rail("popularity", "", "movie", "", 40)
    titles = {m["movie_id"]: (m.get("clean_title") or m.get("title")) for m in candidates}
    options = list(titles.values())
    picked = st.multiselect(
        "Your favorites",
        options,
        max_selections=10,
        placeholder="e.g. Inception, Dangal, Spirited Away ...",
        key="onboard_picks",
        label_visibility="collapsed",
    )
    if st.button("💚 Save my favorites", key="btn_onboard_save", use_container_width=True):
        picked_ids = [mid for mid, t in titles.items() if t in picked]
        errors = 0
        for mid in picked_ids:
            try:
                _post("/api/feedback", {
                    "user_id": st.session_state.user_id,
                    "movie_id": mid,
                    "action": "like",
                })
            except requests.RequestException:
                errors += 1
        st.session_state["onboarding"] = False
        st.session_state.pop("onboard_picks", None)
        if errors:
            st.warning(f"Saved {len(picked_ids) - errors}/{len(picked_ids)} favorites (some failed).")
        else:
            st.toast(f"Saved {len(picked_ids)} favorites — recommendations are now tuned", icon="🎯")
        st.rerun()


def _library_section(uid: int) -> None:
    st.markdown('<div class="cm-rail-title">📚 Your library</div>', unsafe_allow_html=True)
    with st.spinner("Loading your library ..."):
        try:
            lib = _get(f"/api/user/{uid}/library")
        except requests.RequestException as exc:
            st.error(f"Library failed: {exc}")
            return
    liked = lib.get("liked") or []
    watchlist = lib.get("watchlist") or []
    watched = lib.get("watched") or []
    stars = lib.get("stars") or {}
    st.session_state["watched_set"] = {m.get("movie_id") for m in watched}

    if not (liked or watchlist or watched or stars):
        st.markdown(
            '<div class="cm-empty">Your library is empty — 👍 Like, ➕ Watchlist or '
            '⭐ Rate movies to fill it up.</div>',
            unsafe_allow_html=True,
        )
        return

    if liked:
        _rail("💚 Liked", liked, count=len(liked))
    if watchlist:
        _rail("➕ Watchlist", watchlist, count=len(watchlist))
    if watched:
        _rail("✓ Watched", watched, count=len(watched))
    if stars:
        st.markdown('<div class="cm-rail-title">⭐ Your ratings</div>', unsafe_allow_html=True)
        for mid, value in stars.items():
            mid = int(mid)
            movie = next((m for m in (liked + watchlist + watched) if m.get("movie_id") == mid), None)
            title = movie.get("clean_title") or movie.get("title") if movie else f"Movie {mid}"
            st.markdown(
                f"- **{_escape(title)}** &nbsp;<span class='cm-stars'>{_stars(value / 5.0)}</span> "
                f"&nbsp;<span class='cm-score'>{value:.1f}/5</span>",
                unsafe_allow_html=True,
            )


def page_profile() -> None:
    current = st.session_state.get("user", {})
    name = current.get("username", "guest")
    uid = current.get("user_id", DEFAULT_USER)

    if not name or name == "guest":
        _auth_card()
        return

    if st.session_state.get("onboarding"):
        _onboarding_picker()
        return

    st.markdown(
        f'<div class="cm-rail-title">👤 {_escape(name)} · user {uid}</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🙋 Show my profile", key="btn_show_profile", use_container_width=True):
            with st.spinner("Loading profile ..."):
                try:
                    profile = _get(f"/api/user/{uid}/profile")
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
                st.markdown(
                    '<div class="cm-empty">No movie history for this user yet.</div>',
                    unsafe_allow_html=True,
                )
            if watchlist:
                st.markdown(f"**Watchlist ({len(watchlist)})**")
                st.write(", ".join(str(w) for w in watchlist))
    with c2:
        if st.button("🚪 Sign out", key="btn_signout", use_container_width=True):
            st.session_state.user = {"username": "guest", "user_id": DEFAULT_USER}
            st.session_state.user_id = DEFAULT_USER
            st.rerun()
    with c3:
        st.markdown(
            f'<div class="cm-note" style="margin-top:8px;">'
            f'Signed in as **{_escape(name)}** (user {uid}). '
            f'Your 👍 / 👎 / ➕ / ✓ / ⭐ actions on any movie are saved to this account '
            f'and shape your “For You” picks.</div>',
            unsafe_allow_html=True,
        )

    _library_section(uid)


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

    st.session_state.setdefault("user_id", DEFAULT_USER)
    st.session_state.setdefault("user", {"username": "guest", "user_id": DEFAULT_USER})
    st.session_state.setdefault("recommendations", [])
    st.session_state.setdefault("query", "")
    st.session_state.setdefault("page", "Home")

    # A `?detail=<id>` in the URL (clicking a poster tile) deep-links straight
    # into the movie detail page; `?person=<name>` opens the people page. In-app
    # buttons set `session_state.detail` / `session_state.person` directly.
    qp = st.query_params
    if qp.get("detail") is not None:
        val = qp["detail"]
        if isinstance(val, list):
            val = val[0] if val else None
        st.session_state["detail"] = int(val) if str(val).isdigit() else None
        st.session_state.pop("person", None)
    if qp.get("person") is not None:
        val = qp["person"]
        if isinstance(val, list):
            val = val[0] if val else None
        st.session_state["person"] = str(val)
        # Remember which movie we came from so "← Back" can return to it.
        if st.session_state.get("detail") is not None:
            st.session_state["person_back"] = st.session_state["detail"]
        st.session_state.pop("detail", None)

    hd_logo, hd_acc = st.columns([5, 1], vertical_alignment="center")
    with hd_logo:
        st.markdown(
            '<div class="cm-logo">CineMatch<span>.ai</span></div>'
            '<div class="cm-tagline">Recommendations, search & ratings for movies, anime, series and Indian cinema</div>',
            unsafe_allow_html=True,
        )
    with hd_acc:
        _acc = st.session_state.get("user", {})
        _acc_name = _acc.get("username", "guest")
        _acc_label = f"👤 {_escape(_acc_name)}" if _acc_name != "guest" else "🔐 Sign in"
        if st.button(_acc_label, key="btn_account", use_container_width=True, help="Manage your account"):
            st.session_state["nav"] = "Profile"
            st.session_state.pop("detail", None)
            st.session_state.pop("person", None)
            st.query_params.clear()
            st.rerun()

    with st.sidebar:
        st.markdown("### 🔐 Account")
        _acc = st.session_state.get("user", {})
        _acc_name = _acc.get("username", "guest")
        if _acc_name != "guest":
            st.markdown(
                f"Signed in as **{_escape(_acc_name)}** "
                f"(user **{_acc.get('user_id', DEFAULT_USER)}**)."
            )
            if st.button("🚪 Sign out", key="btn_signout_side", use_container_width=True):
                st.session_state.user = {"username": "guest", "user_id": DEFAULT_USER}
                st.session_state.user_id = DEFAULT_USER
                st.rerun()
            st.caption("Manage likes, watchlist and sign-in in the **Profile** tab.")
        else:
            st.markdown(
                "Browsing as **guest**. Sign in from the **Profile** tab to "
                "save your likes and get personal picks."
            )

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
        key="nav",
        label_visibility="collapsed",
    )
    prev_page = st.session_state.get("_last_page")
    st.session_state["_last_page"] = page
    if page is not None:
        if page != prev_page:
            # A nav pill was just clicked — close any open detail / person view.
            st.session_state.pop("detail", None)
            st.session_state.pop("person", None)
            st.query_params.clear()
        st.session_state.page = page

    person_name = st.session_state.get("person")
    if person_name:
        page_person(str(person_name))
        return

    detail_id = st.session_state.get("detail")
    if detail_id is not None:
        page_movie_detail(int(detail_id))
        return

    active = page or st.session_state.page or "Home"
    if active == "Home":
        page_home()
    elif active == "For You":
        page_for_you()
    elif active == "Indian":
        page_indian()
    elif active == "Anime":
        page_anime()
    elif active == "TV":
        page_tv()
    elif active == "Genres":
        page_genres()
    elif active == "Search":
        page_search()
    elif active == "Profile":
        page_profile()


if __name__ == "__main__":
    main()
