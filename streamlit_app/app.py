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

POSTER_CSS = """
<style>
.block-container { padding-top: 2.5rem; max-width: 1500px; }

.hero-title { font-size: 3.6rem; font-weight: 900; letter-spacing: -2px; line-height: 1; margin: 0 0 .7rem 0; }
.hero-title span { color: #E50914; }
.hero-stat { display: inline-block; background: #E50914; font-weight: 800; font-size: .9rem;
             letter-spacing: .3px; padding: 7px 14px; border-radius: 4px; margin-bottom: 1rem; }
.hero-sub { font-size: 1.1rem; color: #9A9A9A; max-width: 780px; line-height: 1.55; }

.row-head { font-size: 1.5rem; font-weight: 800; letter-spacing: -.4px; margin: 0 0 .2rem 0; }
.row-head.generic { color: #7E7E7E; }
.row-head.personal:before { content: ""; display: inline-block; width: 5px; height: 24px;
             background: #E50914; margin-right: 12px; vertical-align: -3px; border-radius: 2px; }
.row-sub { font-size: 1rem; color: #8F8F8F; line-height: 1.5; margin-bottom: 1rem; }

.score { display: inline-block; font-size: .74rem; font-weight: 800; letter-spacing: .5px;
         padding: 4px 9px; border-radius: 3px; vertical-align: 5px; margin-left: 12px;
         background: #262626; color: #8F8F8F; }
.score.best { background: #E50914; color: #FFF; }

.row { display: flex; gap: 14px; overflow-x: auto; padding: 12px 2px 16px 2px; }
.row::-webkit-scrollbar { height: 8px; }
.row::-webkit-scrollbar-track { background: #171717; border-radius: 4px; }
.row::-webkit-scrollbar-thumb { background: #3A3A3A; border-radius: 4px; }
.row:hover::-webkit-scrollbar-thumb { background: #5A5A5A; }

.card { flex: 0 0 168px; }
.poster { position: relative; border-radius: 5px; overflow: hidden; background: #232323;
          aspect-ratio: 2/3; transition: transform .2s ease, box-shadow .2s ease; }
.card:hover .poster { transform: scale(1.07); box-shadow: 0 14px 30px rgba(0,0,0,.8); }
.poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-alt { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
              padding: 12px; text-align: center; font-size: .82rem; color: #6E6E6E; }
.badge { position: absolute; top: 8px; left: 8px; background: #E50914; font-size: .64rem;
         font-weight: 800; letter-spacing: .6px; text-transform: uppercase; padding: 4px 7px; border-radius: 3px; }
.card-title { font-size: .92rem; font-weight: 700; margin-top: 10px; line-height: 1.3; }
.card-genres { font-size: .78rem; color: #7E7E7E; margin-top: 3px; line-height: 1.3; }

.summary { border-left: 5px solid #E50914; background: #161616; padding: 18px 22px;
           border-radius: 4px; font-size: 1.1rem; font-weight: 600; margin-top: 1rem; }
.summary b { color: #E50914; font-size: 1.35rem; }
</style>
"""
st.html(POSTER_CSS)


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


def render_row(movie_ids, unique_ids=None):
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

        art = (
            f'<img src="{poster_url}" alt="{title}">'
            if poster_url
            else f'<div class="poster-alt">{title}</div>'
        )
        badge = (
            '<span class="badge">Unique pick</span>'
            if unique_ids and movie_id in unique_ids
            else ""
        )
        cards.append(
            f'<div class="card"><div class="poster">{art}{badge}</div>'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-genres">{genres}</div></div>'
        )
    st.html(f'<div class="row">{"".join(cards)}</div>')


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


st.html(
    '<div class="hero-title">Cine<span>Match</span></div>'
    '<div class="hero-stat">44.9% better than generic recommendations</div>'
    '<div class="hero-sub">A hybrid recommender that learns taste from behaviour, not just '
    'genre labels — proven against a popularity baseline in a real A/B test (p = 0.00072).</div>'
)

with st.expander("How does this work?"):
    st.write(
        "**Three approaches, side by side.**\n\n"
        "**Genre matching** looks only at what a movie *is* — same genres, same tags. "
        "It produces the most obvious-looking list and, surprisingly, the weakest results.\n\n"
        "**Popularity** ignores you entirely and shows what most people rate highly. "
        "It's a hard baseline to beat precisely because popular films really are widely liked.\n\n"
        "**Our hybrid** is mostly collaborative filtering — it finds people who rate movies "
        "the way you do and recommends what *they* loved, blended with a little genre "
        "signal. It doesn't care what a film is about, which is why its picks can look "
        "unexpected.\n\n"
        "The hit rates come from evaluating each model against ratings it had never seen. "
        "The hybrid's advantage over popularity was confirmed in a real A/B experiment: "
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

pick_col, btn_col, _ = st.columns([2, 1, 2], vertical_alignment="bottom")
with pick_col:
    choice = st.selectbox("Pick a viewer", list(PERSONAS.keys()) + ["Surprise me"])
with btn_col:
    go = st.button("Get recommendations", type="primary", width="stretch")

if go:
    if choice == "Surprise me":
        user_id = random.choice(get_active_users())
        main_genre = get_main_genre(user_id)
    else:
        user_id, main_genre = PERSONAS[choice]
    genre_label = f"{main_genre} movies" if main_genre else "their favourite genre"

    with st.spinner("Finding movies for you..."):
        def fetch(model_name):
            return requests.get(
                f"{API_BASE}/recommend/{user_id}", params={"model_name": model_name}
            ).json()["recommendations"]

        content_ids = fetch("content")
        popularity_ids = fetch("popularity")
        hybrid_ids = fetch("hybrid")

        baseline_ids = set(popularity_ids) | set(content_ids)
        unique_to_hybrid = [m for m in hybrid_ids if m not in baseline_ids]
        hybrid_movies = movies[movies["movieId"].isin(hybrid_ids)]
        surprising = get_surprising_genres(hybrid_movies, main_genre) if main_genre else []

    subject = (
        f"If you like {genre_label}, you"
        if choice == "Surprise me"
        else f"Fans of {genre_label}"
    )
    personal_sub = (
        "We don't just match genres — we match people. "
        + (
            f"{subject} share taste patterns with people who also love "
            f"{' and '.join(surprising)}."
            if surprising
            else f"{subject} often love movies you wouldn't expect."
        )
    )

    st.html(
        '<div class="row-head generic">More of what you already like'
        '<span class="score">5% hit rate</span></div>'
        f'<div class="row-sub">Pure genre matching. You like {genre_label}? Here are '
        f'more {genre_label}. It looks exactly right — and it performs worst of the three.</div>'
    )
    render_row(content_ids)

    st.html(
        '<div class="row-head generic">Everyone\'s favourites'
        '<span class="score">31% hit rate</span></div>'
        '<div class="row-sub">The same list for every viewer, whoever they are — '
        'we just skip what you\'ve already rated.</div>'
    )
    render_row(popularity_ids)

    st.html(
        '<div class="row-head personal">We think you\'ll love these'
        '<span class="score best">44% hit rate</span></div>'
        f'<div class="row-sub">{personal_sub}</div>'
    )
    render_row(hybrid_ids, unique_ids=unique_to_hybrid)

    st.html(
        f'<div class="summary"><b>{len(unique_to_hybrid)} of {len(hybrid_ids)}</b> '
        "personalized picks are movies neither genre matching nor the popularity list "
        "would have shown this viewer.</div>"
    )

st.divider()
st.caption(
    "Both the popularity and personalized recommendations are generated from the "
    "MovieLens (ml-latest-small) research dataset, containing ratings from 1996 to 2018 "
    "— not a live catalog, and not real-time viewer data."
)
