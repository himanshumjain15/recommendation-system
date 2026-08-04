import os
import pandas as pd
from fastapi import FastAPI
from sqlalchemy import create_engine
from dotenv import load_dotenv

from parser.split import build_train_test_split
from scoring.collaborative_filtering import build_cf_model
from scoring.hybrid import build_content_similarity, get_hybrid_recommendations
from scoring.popularity import build_popularity_ranking, recommend_popular

app = FastAPI()

# ---- Runs once, at startup ----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
movies = pd.read_csv(os.path.join(_PROJECT_ROOT, "sample_data/inputs/ml-latest-small/movies.csv"))

ratings, train, held_out = build_train_test_split()
model, user_id_map, movie_id_map, movie_idx_to_id, user_item_matrix = build_cf_model(train)
content_similarity = build_content_similarity(movies)
popularity_ranking = build_popularity_ranking(train)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DATABASE_URL)
# --------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10, model_name: str = "hybrid"):
    if model_name == "popularity":
        recs = recommend_popular(user_id, train, popularity_ranking, k=k)
    else:
        # weight=1.0 is pure content-based (no CF); 0.1 is the tuned hybrid blend
        weight = 1.0 if model_name == "content" else 0.1
        recs = get_hybrid_recommendations(
            user_id, train, movies, model, user_id_map, movie_idx_to_id,
            content_similarity, k=k, weight=weight,
        )

    if recs:
        log_rows = pd.DataFrame({
            "user_id": user_id,
            "movie_id": recs,
            "model_used": model_name,
        })
        log_rows.to_sql("recommendation_logs", engine, if_exists="append", index=False)

    return {"user_id": user_id, "model": model_name, "recommendations": recs}
