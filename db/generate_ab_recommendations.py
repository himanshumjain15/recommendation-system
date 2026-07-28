import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.split import build_train_test_split
from scoring.collaborative_filtering import build_cf_model
from scoring.hybrid import build_content_similarity, get_hybrid_recommendations
from scoring.popularity import build_popularity_ranking, recommend_popular

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DATABASE_URL)

movies = pd.read_csv(os.path.join(_PROJECT_ROOT, "sample_data/inputs/ml-latest-small/movies.csv"))
ratings, train, held_out = build_train_test_split()
all_users = ratings['userId'].unique()

# Build both models once
popularity_ranking = build_popularity_ranking(train)
model, user_id_map, movie_id_map, movie_idx_to_id, user_item_matrix = build_cf_model(train)
content_similarity = build_content_similarity(movies)

# Load each user's assigned group
assignments = pd.read_sql("SELECT user_id, group_name FROM experiment_assignments", engine)
group_by_user = dict(zip(assignments['user_id'], assignments['group_name']))

log_rows = []
for user_id in all_users:
    group = group_by_user.get(user_id)
    if group is None:
        continue

    if group == "control":
        recs = recommend_popular(user_id, train, popularity_ranking, k=10)
        model_used = "popularity"
    else:
        recs = get_hybrid_recommendations(
            user_id, train, movies, model, user_id_map, movie_idx_to_id,
            content_similarity, k=10, weight=0.1,
        )
        model_used = "hybrid"

    for movie_id in recs:
        log_rows.append({"user_id": user_id, "movie_id": movie_id, "model_used": model_used})

log_df = pd.DataFrame(log_rows)
log_df.to_sql("recommendation_logs", engine, if_exists="append", index=False)

print(f"Logged {len(log_df)} recommendations across {log_df['user_id'].nunique()} users")
print(log_df.groupby("model_used")["user_id"].nunique())
