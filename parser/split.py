import os
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_RATINGS_PATH = os.path.join(_PROJECT_ROOT, "sample_data", "inputs", "ml-latest-small", "ratings.csv")


def build_train_test_split(ratings_path=_DEFAULT_RATINGS_PATH):
    """Per-user, time-based 80/20 split. Returns (ratings, train, held_out)."""
    ratings = pd.read_csv(ratings_path)

    ratings_sorted = ratings.sort_values(['userId', 'timestamp']).copy()
    ratings_sorted['rank_in_user'] = ratings_sorted.groupby('userId').cumcount()
    ratings_sorted['user_total'] = ratings_sorted.groupby('userId')['userId'].transform('count')
    ratings_sorted['percentile'] = ratings_sorted['rank_in_user'] / ratings_sorted['user_total']

    train = ratings_sorted[ratings_sorted['percentile'] < 0.8].copy()
    held_out = ratings_sorted[ratings_sorted['percentile'] >= 0.8].copy()

    return ratings, train, held_out
