"""CineMatch AI - Streamlit frontend.

Talk to the FastAPI backend (scripts/run_api.py) via HTTP. Run with:

    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("CINEMATCH_API", "http://localhost:8000")
DEFAULT_USER = int(os.getenv("CINEMATCH_USER", "1"))


def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{API_URL}{path}", json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _post_feedback(action: str, movie_id: int) -> None:
    try:
        _post(
            "/api/feedback",
            {
                "user_id": st.session_state.user_id,
                "movie_id": movie_id,
                "action": action,
            },
        )
        try:
            st.success(f"{action.capitalize()} recorded — recommendations updated")
        except Exception:
            st.write(f"{action.capitalize()} recorded — recommendations updated")
    except requests.RequestException as exc:
        st.error(f"Feedback failed: {exc}")


def get_explore_cards() -> list[dict[str, str]]:
    """Return polished discovery prompts for recruiter-friendly exploration."""
    return [
        {"label": "Mind-bending sci-fi", "query": "smart sci-fi with twists and big ideas", "accent": "🛰️"},
        {"label": "Feel-good classics", "query": "warm-hearted feel-good movies with charm", "accent": "☀️"},
        {"label": "Dark thrillers", "query": "gritty thrillers with suspense and moral tension", "accent": "🌙"},
        {"label": "Global cinema", "query": "international films with strong storytelling and culture", "accent": "🌍"},
        {"label": "Heartfelt dramas", "query": "emotional dramas about relationships and growth", "accent": "💛"},
        {"label": "Action-packed picks", "query": "high-energy action movies with great pacing", "accent": "⚡"},
    ]


def render_movie_card(movie: dict, key_prefix: str) -> None:
    """Render a richer movie card with optional poster image and explanation."""
    title = movie.get("clean_title") or movie.get("title") or "Untitled"
    year = f"({movie.get('year')})" if movie.get("year") else ""
    genres = ", ".join(movie.get("genres", []))
    score = movie.get("score")
    why = movie.get("why")

    cols = st.columns([1, 4])
    left, right = cols

    poster_url = movie.get("poster_url") or movie.get("poster")
    if poster_url:
        try:
            left.image(poster_url, use_column_width=True)
        except Exception:
            left.write("🖼️ Poster not available")
    else:
        left.write(" ")

    with right:
        st.markdown(f"### {title} {year}")
        if genres:
            st.caption(f"*{genres}*")
        if isinstance(score, (int, float)):
            st.write(f"Score: **{score:.3f}**")
        if why:
            source = "LLM" if why.get("source") == "llm" else "rule-based"
            st.info(f"🤖 *Why you might like this ({source}):* {why.get('text')}")


def render_discovery_hero() -> None:
    st.markdown(
        """
        <div style='padding: 1.2rem 1rem; border-radius: 18px; background: linear-gradient(135deg, #0f172a, #312e81); margin-bottom: 1rem;'>
            <h2 style='margin:0; color:white;'>Discover every kind of movie in one neat experience</h2>
            <p style='margin:0.3rem 0 0; color:#dbeafe;'>From indie gems to blockbuster hits, this experience feels polished enough for a recruiter demo and smart enough for real movie lovers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_cols = st.columns([2, 1])
    with hero_cols[0]:
        st.write("The interface blends recommendation intelligence with cinematic discovery, so visitors can browse by mood, taste, and intent in seconds.")
        st.caption("Built to impress with a premium feel, fast interactions, and broad catalog coverage.")
    with hero_cols[1]:
        st.metric("Catalog breadth", "All major movie moods")
        st.metric("Experience", "Hybrid search + recommendations")


def render_explore_cards() -> None:
    cards = get_explore_cards()
    cols = st.columns(3)
    for idx, card in enumerate(cards):
        with cols[idx % 3]:
            if st.button(f"{card['accent']} {card['label']}", key=f"explore_{idx}", use_container_width=True):
                st.session_state.search_query = card["query"]
                st.session_state.active_discovery = card["label"]

    if st.session_state.get("active_discovery"):
        st.success(f"Showing discovery results for: {st.session_state.active_discovery}")


def main() -> None:
    st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem}
        h1 {text-align: center; margin-bottom: 0.1rem}
        .stCaption {text-align: center; margin-top: 0; color: #64748b}
        .stButton>button {border-radius: 999px; padding: 0.5rem 1rem}
        .stTabs [data-baseweb="tab-list"] {gap: 0.4rem}
        .stTabs [data-baseweb="tab"] {border-radius: 999px; padding: 0.5rem 0.9rem}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🎬 CineMatch AI")
    st.caption("Hybrid movie recommendations with cinematic discovery for every taste")

    st.session_state.setdefault("user_id", DEFAULT_USER)
    st.session_state.setdefault("recommendations", [])
    st.session_state.setdefault("query", "")
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("active_discovery", "")

    DEMO_RECOMMENDATIONS = [
        {
            "movie_id": 1,
            "clean_title": "The Demo Odyssey",
            "title": "The Demo Odyssey",
            "year": 2022,
            "genres": ["Sci-Fi", "Adventure"],
            "score": 0.97,
            "why": {"source": "llm", "text": "Stunning visuals and an emotional core."},
            "poster_url": "https://via.placeholder.com/300x450.png?text=Demo+Poster",
        },
        {
            "movie_id": 2,
            "clean_title": "Hackers' Ball",
            "title": "Hackers' Ball",
            "year": 2019,
            "genres": ["Thriller"],
            "score": 0.88,
            "why": {"source": "rule", "text": "Fast-paced plot, great for late-night viewing."},
            "poster_url": "https://via.placeholder.com/300x450.png?text=Demo+Poster",
        },
    ]

    with st.sidebar:
        st.header("Session")
        st.session_state.user_id = st.number_input(
            "User ID", min_value=1, value=st.session_state.user_id, step=1
        )
        try:
            health = _get("/health")
            st.success(
                f"API ok · {health['movies']:,} movies · "
                f"vectors: {'indexed' if health['vector_index'] else 'NOT indexed'}"
            )
            if not health["vector_index"]:
                st.warning("Run `python scripts/index_vectors.py` to enable semantic search.")
        except requests.RequestException:
            st.error(f"Cannot reach API at {API_URL}. Start it with `python scripts/run_api.py`.")

        use_demo = st.checkbox("Use demo data (no backend)", value=False)
        if use_demo:
            st.info("Using built-in demo data — good for showing the UI to recruiters without running the backend.")

        if st.button("Show my profile"):
            with st.spinner("Loading profile ..."):
                profile = _get(f"/api/user/{st.session_state.user_id}/profile")
                st.subheader(f"Top liked ({len(profile['liked'])})")
                for m in profile["liked"][:8]:
                    st.write(f"- {m['title']} ({m['rating']:.1f})")

    tab_discover, tab_recs, tab_search = st.tabs(["✨ Discover", "🔮 Recommendations", "🔎 Semantic search"])

    with tab_discover:
        render_discovery_hero()
        st.subheader("Browse by mood")
        render_explore_cards()
        if st.session_state.get("search_query"):
            with st.spinner("Searching the catalog ..."):
                try:
                    results = _get("/api/movies/search", {"q": st.session_state.search_query, "n": 6})["results"]
                except requests.RequestException:
                    results = []
                    st.warning("The backend is not available right now, so only the curated discovery prompts are shown.")
            if results:
                st.subheader("Results for this vibe")
                for movie in results:
                    render_movie_card(movie, key_prefix=f"discover_{movie.get('movie_id')}")

    with tab_recs:
        col_btn, col_query = st.columns([1, 3])
        with col_btn:
            if st.button("Get recommendations"):
                with st.spinner("Running hybrid pipeline ..."):
                    if use_demo if "use_demo" in locals() else False:
                        st.session_state.recommendations = DEMO_RECOMMENDATIONS
                    else:
                        st.session_state.recommendations = _get(
                            f"/api/recommend/{st.session_state.user_id}",
                            {"n": 10, "with_explanations": True},
                        )["recommendations"]
        with col_query:
            st.session_state.query = st.text_input(
                "Optional natural-language preference (e.g. 'movies like Interstellar but focused on deep sea exploration')",
                value=st.session_state.query,
                placeholder="Describe what you feel like watching ...",
            )
            if st.button("Recommend with preference"):
                with st.spinner("Blending query into recommendations ..."):
                    st.session_state.recommendations = _post(
                        "/api/recommend/query",
                        {
                            "user_id": st.session_state.user_id,
                            "query": st.session_state.query,
                            "n": 10,
                            "with_explanations": True,
                        },
                    )["recommendations"]

        for i, movie in enumerate(st.session_state.recommendations):
            with st.container():
                render_movie_card(movie, key_prefix=f"rec_{i}")
                c1, c2, c3, c4 = st.columns(4)
                mid = movie.get("movie_id")
                with c1:
                    if st.button("👍 Like", key=f"like_{i}"):
                        _post_feedback("like", mid)
                with c2:
                    if st.button("👎 Dislike", key=f"dis_{i}"):
                        _post_feedback("dislike", mid)
                with c3:
                    if st.button("➕ Watchlist", key=f"wl_{i}"):
                        _post_feedback("watchlist", mid)
                with c4:
                    if st.button("🧹 Reset", key=f"rm_{i}"):
                        _post_feedback("remove", mid)

    with tab_search:
        query = st.text_input("Search in natural language", placeholder="gritty 90s thriller with a mind-bending plot twist")
        if st.button("Search", type="primary"):
            if query.strip():
                with st.spinner("Searching vectors ..."):
                    try:
                        results = _get("/api/movies/search", {"q": query, "n": 10})["results"]
                    except requests.RequestException:
                        results = []
                        st.warning("The backend is not available right now.")
                for movie in results:
                    with st.container():
                        render_movie_card(movie, key_prefix=f"search_{movie.get('movie_id')}")


if __name__ == "__main__":
    main()
