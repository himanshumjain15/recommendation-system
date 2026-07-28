import os
import sys
import hashlib
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser.split import build_train_test_split


def bucket_user(user_id):
    """Deterministic, well-mixed assignment to control/treatment (unlike hash(int) % 2,
    which just checks even/odd, since Python's built-in hash() of an int is itself)."""
    digest = hashlib.md5(str(user_id).encode()).hexdigest()
    return "control" if int(digest, 16) % 2 == 0 else "treatment"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DATABASE_URL)

ratings, train, held_out = build_train_test_split()
all_users = ratings['userId'].unique()

assignments = pd.DataFrame({
    "user_id": all_users,
    "group_name": [bucket_user(user_id) for user_id in all_users],
})

assignments.to_sql("experiment_assignments", engine, if_exists="append", index=False)

print(assignments["group_name"].value_counts())
print(f"Total assigned: {len(assignments)}")
