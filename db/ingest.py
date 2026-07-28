import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

engine = create_engine(DATABASE_URL)

ratings = pd.read_csv(os.path.join(_PROJECT_ROOT, "sample_data/inputs/ml-latest-small/ratings.csv"))
movies = pd.read_csv(os.path.join(_PROJECT_ROOT, "sample_data/inputs/ml-latest-small/movies.csv"))

# users: one row per distinct userId
users_df = pd.DataFrame({"user_id": ratings["userId"].unique()})
users_df.to_sql("users", engine, if_exists="append", index=False)
print(f"Inserted {len(users_df)} users")

# items: movies.csv, renamed to match schema
items_df = movies.rename(columns={"movieId": "movie_id"})[["movie_id", "title", "genres"]]
items_df.to_sql("items", engine, if_exists="append", index=False)
print(f"Inserted {len(items_df)} items")

# interactions: ratings.csv, renamed + timestamp converted to a real datetime
interactions_df = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})[
    ["user_id", "movie_id", "rating"]
]
interactions_df["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s")
interactions_df.to_sql("interactions", engine, if_exists="append", index=False)
print(f"Inserted {len(interactions_df)} interactions")
