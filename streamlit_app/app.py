import io
import os
import random
import zipfile

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "sample_data", "inputs", "ml-latest-small")
_DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


def _secret(name, default=None):
    """Read config from Streamlit secrets when deployed, .env when running locally."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:
        pass
    return os.getenv(name, default)


API_BASE = _secret("API_BASE", "http://3.134.153.110:8000")
TMDB_API_KEY = _secret("TMDB_API_KEY")

POSTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800;900&display=swap');

.block-container { padding-top: 2.5rem; max-width: 1500px; }

.hero { position: relative; overflow: hidden; border-radius: 8px;
        padding: 3.2rem 2.8rem 2.8rem 2.8rem; margin-bottom: 1.8rem; background: #0B0B0B; }
.hero-bg { position: absolute; inset: 0; display: flex; opacity: .30; }
.hero-bg img { flex: 1 1 0; min-width: 0; height: 100%; object-fit: cover; }
.hero:after { content: ""; position: absolute; inset: 0;
              background: linear-gradient(90deg, #0B0B0B 18%, rgba(11,11,11,.86) 52%, rgba(11,11,11,.62) 100%); }
.hero-fg { position: relative; z-index: 2; }

.hero-title { font-family: 'Archivo', sans-serif; font-size: 4rem; font-weight: 900;
              letter-spacing: -2.5px; line-height: 1.12; margin: 0 0 .9rem 0;
              text-transform: uppercase; }
.hero-title span { color: #E50914; }
.hero-stat { display: inline-block; background: #E50914; font-weight: 800; font-size: .9rem;
             letter-spacing: .3px; padding: 7px 14px; border-radius: 4px; margin-bottom: 1rem; }
.hero-sub { font-size: 1.12rem; color: #C4C4C4; max-width: 660px; line-height: 1.55; }

.row-head { font-family: 'Archivo', sans-serif; font-size: 1.55rem; font-weight: 800;
            letter-spacing: -.4px; margin: 0 0 .2rem 0; }
.row-head.generic { color: #7E7E7E; }
.row-head.personal:before { content: ""; display: inline-block; width: 5px; height: 24px;
             background: #E50914; margin-right: 12px; vertical-align: -3px; border-radius: 2px; }
.row-sub { font-size: 1rem; color: #8F8F8F; line-height: 1.5; margin-bottom: 1rem; }

.score { display: inline-block; font-size: .74rem; font-weight: 800; letter-spacing: .5px;
         padding: 4px 9px; border-radius: 3px; vertical-align: 5px; margin-left: 12px;
         background: #262626; color: #8F8F8F; }
.score.best { background: #E50914; color: #FFF; }

.row-wrap { position: relative; }
.row-wrap:before, .row-wrap:after { content: ""; position: absolute; top: 0; bottom: 18px;
            width: 48px; pointer-events: none; z-index: 3; }
.row-wrap:before { left: 0; background: linear-gradient(90deg, #0B0B0B 10%, transparent); }
.row-wrap:after { right: 0; background: linear-gradient(270deg, #0B0B0B 10%, transparent); }

.row { display: flex; gap: 12px; overflow-x: auto; padding: 14px 2px 18px 2px; }
.row::-webkit-scrollbar { height: 8px; }
.row::-webkit-scrollbar-track { background: #171717; border-radius: 4px; }
.row::-webkit-scrollbar-thumb { background: #3A3A3A; border-radius: 4px; }
.row:hover::-webkit-scrollbar-thumb { background: #5A5A5A; }

.card { flex: 0 0 172px; }
.poster { position: relative; border-radius: 5px; overflow: hidden; background: #232323;
          aspect-ratio: 2/3; transition: transform .22s ease, box-shadow .22s ease; }
.card:hover .poster { transform: scale(1.08); box-shadow: 0 16px 34px rgba(0,0,0,.85); z-index: 4; }
.poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-alt { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
              padding: 12px; text-align: center; font-size: .82rem; color: #8A8A8A; }
.badge { position: absolute; top: 8px; left: 8px; background: #E50914; font-size: .64rem;
         font-weight: 800; letter-spacing: .6px; text-transform: uppercase; padding: 4px 7px;
         border-radius: 3px; z-index: 2; }

.meta { position: absolute; left: 0; right: 0; bottom: 0; padding: 40px 11px 11px 11px;
        background: linear-gradient(transparent, rgba(0,0,0,.55) 40%, rgba(0,0,0,.94));
        opacity: 0; transform: translateY(10px); transition: opacity .22s ease, transform .22s ease; }
.card:hover .meta { opacity: 1; transform: none; }
.meta-title { font-size: .88rem; font-weight: 700; line-height: 1.25; }
.meta-genres { font-size: .73rem; color: #B8B8B8; margin-top: 4px; line-height: 1.25; }
@media (hover: none) { .meta { opacity: 1; transform: none; } }

.summary { border-left: 5px solid #E50914; background: #161616; padding: 18px 22px;
           border-radius: 4px; font-size: 1.1rem; font-weight: 600; margin-top: 1rem; }
.summary b { color: #E50914; font-size: 1.35rem; }

.approach-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                 gap: 14px; margin: .2rem 0 1.4rem 0; }
.approach { background: #131313; border: 1px solid #272727; border-radius: 6px; padding: 18px 20px; }
.approach.win { border-color: #E50914; }
.approach-name { font-family: 'Archivo', sans-serif; font-size: 1.02rem; font-weight: 800;
                 letter-spacing: -.2px; }
.approach-score { font-family: 'Archivo', sans-serif; font-size: 2.3rem; font-weight: 900;
                  line-height: 1.15; color: #6E6E6E; }
.approach.win .approach-score { color: #E50914; }
.approach-unit { font-size: .74rem; color: #7E7E7E; margin-bottom: 11px; }
.approach-bar { height: 5px; background: #262626; border-radius: 3px; overflow: hidden;
                margin-bottom: 13px; }
.approach-bar span { display: block; height: 100%; background: #4E4E4E; }
.approach.win .approach-bar span { background: #E50914; }
.approach-desc { font-size: .87rem; color: #A6A6A6; line-height: 1.55; }

.taste { display: flex; align-items: center; gap: 30px; flex-wrap: wrap;
         background: #131313; border: 1px solid #232323; border-radius: 6px;
         padding: 16px 22px; margin: .3rem 0 1.6rem 0; }
.taste-label { flex: 0 0 100%; font-size: .72rem; text-transform: uppercase;
               letter-spacing: .9px; color: #6C6C6C; }
.taste-lead { font-size: .78rem; color: #8A8A8A; line-height: 1.4; white-space: nowrap; }
.taste-lead b { display: block; font-family: 'Archivo', sans-serif; font-size: 1.7rem;
                font-weight: 900; color: #F5F5F5; line-height: 1.1; }
.taste-bars { display: flex; gap: 20px; flex: 1; min-width: 0; }
.taste-item { flex: 1; min-width: 0; }
.taste-top { display: flex; justify-content: space-between; gap: 8px; font-size: .78rem;
             margin-bottom: 6px; }
.taste-name { color: #D2D2D2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.taste-pct { color: #7C7C7C; }
.taste-track { height: 6px; background: #262626; border-radius: 3px; overflow: hidden; }
.taste-fill { height: 100%; background: #E50914; border-radius: 3px; }

.footer { border-top: 1px solid #242424; margin-top: 3rem; padding-top: 1.6rem;
          display: flex; flex-wrap: wrap; align-items: baseline; gap: 14px 34px; }
.footer-main { flex: 1; min-width: 220px; }
.footer-name { font-family: 'Archivo', sans-serif; font-size: 1.05rem; font-weight: 800;
               color: #EDEDED; }
.footer-role { font-size: .82rem; color: #7A7A7A; margin-top: 2px; }
.footer-links { display: flex; gap: 22px; }
.footer-links a { font-size: .88rem; font-weight: 600; color: #C8C8C8;
                  text-decoration: none; border-bottom: 1px solid #3A3A3A; padding-bottom: 2px; }
.footer-links a:hover { color: #E50914; border-bottom-color: #E50914; }
.footer-note { flex: 0 0 100%; font-size: .76rem; color: #656565; line-height: 1.6;
               margin-top: .4rem; }
</style>
"""
st.html(POSTER_CSS)


@st.cache_resource(show_spinner="Fetching the MovieLens dataset...")
def ensure_dataset():
    """Download the dataset on first run. Locally it's usually already on disk; on a
    fresh deployment it isn't, since the raw data is deliberately not committed."""
    if os.path.exists(os.path.join(_DATA_DIR, "movies.csv")):
        return _DATA_DIR
    target = os.path.dirname(_DATA_DIR)
    os.makedirs(target, exist_ok=True)
    archive = requests.get(_DATA_URL, timeout=60)
    archive.raise_for_status()
    zipfile.ZipFile(io.BytesIO(archive.content)).extractall(target)
    return _DATA_DIR


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(ensure_dataset(), name))


movies = load_csv("movies.csv")
ratings = load_csv("ratings.csv")
links = load_csv("links.csv")

if not TMDB_API_KEY:
    st.warning(
        "No TMDB API key found, so posters will fall back to titles. Set `TMDB_API_KEY` "
        "in `.env` locally, or in the app's secrets when deployed.",
        icon=":material/image_not_supported:",
    )


@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id):
    if not TMDB_API_KEY or pd.isna(tmdb_id):
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


@st.cache_data
def get_taste_profile(user_id, top_n=5):
    """How many films this viewer rated, and the genre mix of the ones they loved."""
    rated = ratings[ratings["userId"] == user_id]
    loved = rated[rated["rating"] >= 4].merge(movies, on="movieId")
    if loved.empty:
        return len(rated), []
    shares = loved["genres"].str.split("|").explode().value_counts() / len(loved) * 100
    return len(rated), [(genre, round(pct)) for genre, pct in shares.head(top_n).items()]


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
            f'<div class="card"><div class="poster">{art}{badge}'
            f'<div class="meta"><div class="meta-title">{title}</div>'
            f'<div class="meta-genres">{genres}</div></div>'
            f'</div></div>'
        )
    st.html(f'<div class="row-wrap"><div class="row">{"".join(cards)}</div></div>')


def get_main_genre(user_id):
    liked = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]
    liked = liked.merge(movies, on="movieId")
    genre_counts = liked["genres"].str.split("|").explode().value_counts()
    return genre_counts.index[0] if len(genre_counts) else None


def get_surprising_genres(recommended_movies, known_genres, top_n=2):
    """Genres in the recommendations that aren't already staples of this viewer's taste."""
    genre_counts = recommended_movies["genres"].str.split("|").explode().value_counts()
    genre_counts = genre_counts.drop(labels=list(known_genres), errors="ignore")
    return list(genre_counts.head(top_n).index)


@st.cache_data
def get_active_users(min_ratings=20):
    counts = ratings.groupby("userId").size()
    return counts[counts >= min_ratings].index.tolist()


@st.cache_data(show_spinner=False)
def hero_backdrop(n=9):
    """Posters of the most-rated films, used as a darkened backdrop behind the title."""
    urls = []
    for movie_id in ratings["movieId"].value_counts().head(40).index:
        link_row = links[links["movieId"] == movie_id]
        if link_row.empty:
            continue
        url = get_poster_url(link_row["tmdbId"].values[0])
        if url:
            urls.append(url)
        if len(urls) == n:
            break
    return urls


backdrop = "".join(f'<img src="{u}">' for u in hero_backdrop())
st.html(
    f'<div class="hero"><div class="hero-bg">{backdrop}</div><div class="hero-fg">'
    '<div class="hero-title">Cine<span>Match</span></div>'
    '<div class="hero-stat">Lands a hit for 44 viewers in 100. Popularity manages 31</div>'
    '<div class="hero-sub">A hybrid recommender that learns taste from behaviour rather '
    'than genre labels, proven against a popularity baseline in a real A/B test '
    '(p = 0.00072).</div>'
    '</div></div>'
)

APPROACHES = [
    ("Our hybrid", 44, True,
     "Finds people who liked the same films as this viewer, and recommends what "
     "<em>they</em> loved, with a little genre signal mixed in. It doesn't care what a "
     "film is about, which is why its picks can look unexpected."),
    ("Popularity", 31, False,
     "Ignores the viewer entirely and shows what most people rate highly. A hard baseline "
     "to beat, because popular films really are widely liked."),
    ("Genre matching", 5, False,
     "Looks only at what a film <em>is</em>: same genres, same tags. Produces the most "
     "obvious-looking list and, surprisingly, the weakest results."),
]

with st.expander("How does this work?"):
    cards = "".join(
        f'<div class="approach{" win" if win else ""}">'
        f'<div class="approach-name">{name}</div>'
        f'<div class="approach-score">{score}</div>'
        f'<div class="approach-unit">viewers in 100</div>'
        f'<div class="approach-bar"><span style="width:{round(score / 44 * 100)}%"></span></div>'
        f'<div class="approach-desc">{desc}</div></div>'
        for name, score, win, desc in APPROACHES
    )
    st.html(f'<div class="approach-grid">{cards}</div>')
    st.write(
        "**How those numbers were measured:** we hid part of each viewer's real rating "
        "history from the model, then checked whether its recommendations included "
        "something that viewer went on to rate highly. The hybrid managed that for 44 "
        "viewers in 100; pure genre matching managed it for 5.\n\n"
        "The hybrid's advantage over popularity was confirmed in a real A/B experiment: "
        "a 44.9% lift, statistically significant at p = 0.00072. "
        "[See the full write-up on GitHub](https://github.com/himanshumjain15/recommendation-system)."
    )

PERSONAS = {
    ":material/bolt: The action fan": (380, "Action"),
    ":material/family_restroom: The family viewer": (20, "Animation"),
    ":material/mood: The comedy lover": (89, "Comedy"),
    ":material/theaters: The drama devotee": (74, "Drama"),
    ":material/favorite: The romantic": (594, "Romance"),
    ":material/rocket_launch: The sci-fi fan": (186, "Sci-Fi"),
    ":material/visibility: The thriller seeker": (80, "Thriller"),
}

def use_persona():
    """Picking from the dropdown cancels any random viewer."""
    st.session_state.pop("random_user", None)


pick_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
with pick_col:
    choice = st.pills(
        "Pick a viewer",
        list(PERSONAS.keys()),
        default=list(PERSONAS.keys())[0],
        required=True,
        on_change=use_persona,
    )
with btn_col:
    if st.button(":material/casino: Surprise me", width="stretch"):
        st.session_state["random_user"] = random.choice(get_active_users())

if "random_user" in st.session_state:
    user_id = st.session_state["random_user"]
    main_genre = get_main_genre(user_id)
else:
    user_id, main_genre = PERSONAS[choice]
genre_label = f"{main_genre} movies" if main_genre else "their favourite genre"
genre_short = main_genre if main_genre else "a genre"

rated_count, taste = get_taste_profile(user_id)
if taste:
    bars = "".join(
        f'<div class="taste-item"><div class="taste-top">'
        f'<span class="taste-name">{genre}</span><span class="taste-pct">{pct}%</span></div>'
        f'<div class="taste-track"><div class="taste-fill" style="width:{pct}%"></div></div>'
        f"</div>"
        for genre, pct in taste
    )
    st.html(
        '<div class="taste">'
        '<div class="taste-label">What this viewer actually loves</div>'
        f'<div class="taste-lead">Rated<b>{rated_count}</b>films</div>'
        f'<div class="taste-bars">{bars}</div></div>'
    )

with st.spinner("Finding films..."):
    def fetch(model_name):
        response = requests.get(
            f"{API_BASE}/recommend/{user_id}",
            params={"model_name": model_name},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["recommendations"]

    try:
        content_ids = fetch("content")
        popularity_ids = fetch("popularity")
        hybrid_ids = fetch("hybrid")
    except requests.exceptions.RequestException:
        st.error(
            "**The recommendation service isn't responding.**\n\n"
            "This demo calls a live API running on a single AWS instance, which is "
            "sometimes switched off to keep costs down. The code and full write-up are "
            "on [GitHub](https://github.com/himanshumjain15/recommendation-system) "
            "either way.",
            icon=":material/cloud_off:",
        )
        st.stop()

    baseline_ids = set(popularity_ids) | set(content_ids)
    unique_to_hybrid = [m for m in hybrid_ids if m not in baseline_ids]
    hybrid_movies = movies[movies["movieId"].isin(hybrid_ids)]
    known_genres = {genre for genre, _ in taste[:3]} | ({main_genre} if main_genre else set())
    surprising = get_surprising_genres(hybrid_movies, known_genres)

personal_sub = (
    f"Not just more {genre_label}. People who liked the same films as this viewer also "
    f"loved {' and '.join(surprising)}."
    if surprising
    else "Drawn from people who liked the same films as this viewer, not from what the "
    "films happen to be about."
)

st.html(
    '<div class="row-head personal">We think you\'ll love these'
    '<span class="score best">works for 44 in 100</span></div>'
    f'<div class="row-sub">{personal_sub}</div>'
)
render_row(hybrid_ids, unique_ids=unique_to_hybrid)

st.html(
    '<div class="row-head generic">Everyone\'s favourites'
    '<span class="score">works for 31 in 100</span></div>'
    '<div class="row-sub">Ranked purely by how many people rated each film. The same '
    "ranking for everyone, minus whatever this viewer has already rated. Heavy raters "
    'have seen the famous ones, so what surfaces here differs.</div>'
)
render_row(popularity_ids)

st.html(
    '<div class="row-head generic">More of the same genre'
    '<span class="score">works for 5 in 100</span></div>'
    f'<div class="row-sub">This viewer likes {genre_short}, so here is more '
    f'{genre_short}. The obvious approach, and the one that works least often.</div>'
)
render_row(content_ids)

st.html(
    f'<div class="summary"><b>{len(unique_to_hybrid)} of {len(hybrid_ids)}</b> '
    "personalized picks are movies neither genre matching nor the popularity list "
    "would have shown this viewer.</div>"
)

st.html(
    '<div class="footer">'
    '<div class="footer-main">'
    '<div class="footer-name">Built by Himanshu Jain</div>'
    '<div class="footer-role">MS Data Science, University of Colorado Boulder</div>'
    "</div>"
    '<div class="footer-links">'
    '<a href="https://github.com/himanshumjain15/recommendation-system">Source code</a>'
    '<a href="https://himanshumjain15.github.io/Portfolio/">Portfolio</a>'
    f'<a href="{API_BASE}/docs">API</a>'
    "</div>"
    '<div class="footer-note">'
    "Recommendations come from the MovieLens (ml-latest-small) research dataset, "
    "ratings collected between 1996 and 2018. Not a live catalogue, and not real-time "
    "viewer data. Posters courtesy of TMDB, which did not endorse this project."
    "</div></div>"
)
