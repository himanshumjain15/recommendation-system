import os
import random
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

API_BASE = "http://3.134.153.110:8000"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

CSS = """
<style>
.block-container { padding-top: 2.5rem; max-width: 1400px; }

.hero-title {
    font-size: 3.4rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    line-height: 1.05;
    margin: 0 0 0.6rem 0;
}
.hero-title span { color: #E50914; }
.hero-sub {
    font-size: 1.15rem;
    color: #B3B3B3;
    max-width: 720px;
    line-height: 1.5;
    margin-bottom: 0.4rem;
}
.hero-stat {
    display: inline-block;
    background: #E50914;
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 6px 14px;
    border-radius: 4px;
    margin-bottom: 1.2rem;
}

.col-header {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.2rem 0 0.3rem 0;
    letter-spacing: -0.4px;
}
.col-header.generic { color: #8C8C8C; }
.col-header.personal { color: #FFFFFF; }
.col-header.personal:before {
    content: "";
    display: inline-block;
    width: 5px; height: 26px;
    background: #E50914;
    margin-right: 12px;
    vertical-align: -4px;
    border-radius: 2px;
}
.col-sub {
    font-size: 1rem;
    color: #A0A0A0;
    line-height: 1.5;
    margin-bottom: 1.4rem;
    min-height: 76px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(135px, 1fr));
    gap: 22px 16px;
}
.card { position: relative; }
.poster-wrap {
    position: relative;
    border-radius: 6px;
    overflow: hidden;
    background: #2A2A2A;
    aspect-ratio: 2 / 3;
    transition: transform .18s ease, box-shadow .18s ease;
}
.card:hover .poster-wrap {
    transform: scale(1.06);
    box-shadow: 0 12px 28px rgba(0,0,0,.75);
}
.poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-fallback {
    width: 100%; height: 100%;
    display: flex; align-items: center; justify-content: center;
    padding: 10px; text-align: center;
    font-size: .8rem; color: #7A7A7A;
}
.badge {
    position: absolute; top: 8px; left: 8px;
    background: #E50914; color: #fff;
    font-size: .68rem; font-weight: 800;
    letter-spacing: .5px; text-transform: uppercase;
    padding: 4px 8px; border-radius: 3px;
}
.card-title {
    font-size: .95rem; font-weight: 700;
    margin-top: 10px; line-height: 1.3;
}
.card-genres {
    font-size: .8rem; color: #8C8C8C;
    margin-top: 3px; line-height: 1.35;
}
.summary {
    border-left: 5px solid #E50914;
    background: #1A1A1A;
    padding: 18px 22px;
    border-radius: 4px;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 2.5rem;
}
.summary b { color: #E50914; font-size: 1.35rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(_PROJECT_ROOT, f"sample_data/inputs/ml-latest-small/{name}"))


movies = load_csv("movies.csv")
ratings = load_csv("ratings.csv")
links = load_csv("links.csv")


@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id):
    if pd.isna(tmdb_id):
        return None
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        poster_path = response.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None
    except requests.exceptions.RequestException:
        return None


def render_grid(movie_ids, unique_ids=None):
    cards = []
    for movie_id in movie_ids:
        row = movies[movies["movieId"] == movie_id]
        if row.empty:
            continue
        title = row["title"].values[0]
        genres = row["genres"].values[0].replace("|", " · ")

        link_row = links[links["movieId"] == movie_id]
        tmdb_id = link_row["tmdbId"].values[0] if not link_row.empty else None
        poster_url = get_poster_url(tmdb_id)

        img = (
            f'<img src="{poster_url}" alt="{title}">'
            if poster_url
            else f'<div class="poster-fallback">{title}</div>'
        )
        badge = (
            '<span class="badge">Unique pick</span>'
            if unique_ids and movie_id in unique_ids
            else ""
        )
        cards.append(
            f'<div class="card"><div class="poster-wrap">{img}{badge}</div>'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-genres">{genres}</div></div>'
        )
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def get_main_genre(user_id):
    liked = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]
    liked = liked.merge(movies, on="movieId")
    genre_counts = liked["genres"].str.split("|").explode().value_counts()
    return genre_counts.index[0] if len(genre_counts) else None


def get_surprising_genres(recommended_movies, main_genre, top_n=2):
    genre_counts = recommended_movies["genres"].str.split("|").explode().value_counts()
    genre_counts = genre_counts.drop(labels=[main_genre], errors="ignore")
    return list(genre_counts.head(top_n).index)


@st.cache_data
def get_active_users(min_ratings=20):
    counts = ratings.groupby("userId").size()
    return counts[counts >= min_ratings].index.tolist()


st.markdown(
    '<div class="hero-title">Cine<span>Match</span></div>'
    '<div class="hero-stat">44.9% better than generic recommendations</div>'
    '<div class="hero-sub">A hybrid recommender that learns taste from behaviour, not just '
    'genre labels — proven against a popularity baseline in a real A/B test (p = 0.00072).</div>',
    unsafe_allow_html=True,
)

with st.expander("How does this work?"):
    st.write(
        "**Two very different approaches.** The left side just shows what's popular with "
        "everyone — no personalization at all. The right side blends two techniques: one "
        "looks at what a movie is about (genre), the other looks at people — who rates "
        "movies the way you do, and what they loved. That second one is collaborative "
        "filtering, and it's the real engine behind your personalized picks.\n\n"
        "We put this to the test against the popularity approach in a real A/B experiment: "
        "a 44.9% lift, statistically significant at p = 0.00072. "
        "[See the full write-up on GitHub](https://github.com/himanshumjain15/recommendation-system)."
    )

PERSONAS = {
    "The Action Fan": (380, "Action"),
    "The Animation & Family Fan": (20, "Animation"),
    "The Comedy Enthusiast": (414, "Comedy"),
    "The Romance Lover": (594, "Romance"),
    "The Sci-Fi Fan": (186, "Sci-Fi"),
    "The Thriller Fan": (610, "Thriller"),
}

pick_col, btn_col, _ = st.columns([2, 1, 2])
with pick_col:
    choice = st.selectbox("Pick a viewer", list(PERSONAS.keys()) + ["Surprise me"])
with btn_col:
    st.write("")
    st.write("")
    go = st.button("Get recommendations", type="primary", use_container_width=True)

if go:
    if choice == "Surprise me":
        user_id = random.choice(get_active_users())
        main_genre = get_main_genre(user_id)
    else:
        user_id, main_genre = PERSONAS[choice]
    genre_label = f"{main_genre} movies" if main_genre else "their favourite genre"

    with st.spinner("Finding movies for you..."):
        hybrid_ids = requests.get(
            f"{API_BASE}/recommend/{user_id}", params={"model_name": "hybrid"}
        ).json()["recommendations"]
        popularity_ids = requests.get(
            f"{API_BASE}/recommend/{user_id}", params={"model_name": "popularity"}
        ).json()["recommendations"]

        unique_to_hybrid = [m for m in hybrid_ids if m not in popularity_ids]
        hybrid_movies = movies[movies["movieId"].isin(hybrid_ids)]
        surprising = get_surprising_genres(hybrid_movies, main_genre) if main_genre else []

    subject = (
        f"If you like {genre_label}, you"
        if choice == "Surprise me"
        else f"Fans of {genre_label}"
    )
    if surprising:
        personal_sub = (
            "We don't just match genres — we match people. "
            f"{subject} share taste patterns with people who also love "
            f"{' and '.join(surprising)}."
        )
    else:
        personal_sub = (
            "We don't just match genres — we match people. "
            f"{subject} often love movies you wouldn't expect."
        )

    st.write("")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="col-header generic">Everyone\'s favourites</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="col-sub">The same list for every viewer, whoever they are — '
            'we just skip what you\'ve already rated.</div>',
            unsafe_allow_html=True,
        )
        render_grid(popularity_ids)

    with col2:
        st.markdown('<div class="col-header personal">We think you\'ll love these</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="col-sub">{personal_sub}</div>', unsafe_allow_html=True)
        render_grid(hybrid_ids, unique_ids=unique_to_hybrid)

    st.markdown(
        f'<div class="summary"><b>{len(unique_to_hybrid)} of {len(hybrid_ids)}</b> '
        "personalized picks are movies the generic popularity list would never have "
        "shown this viewer.</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Both the popularity and personalized recommendations are generated from the "
    "MovieLens (ml-latest-small) research dataset, containing ratings from 1996 to 2018 "
    "— not a live catalog, and not real-time viewer data."
)
