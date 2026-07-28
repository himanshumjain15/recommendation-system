def build_popularity_ranking(train):
    """Ranks movies by number of ratings in train, most-rated first."""
    return train['movieId'].value_counts()


def recommend_popular(user_id, train, popularity_ranking, k=10):
    """Top-k most popular movies this user hasn't already rated."""
    already_rated = train[train['userId'] == user_id]['movieId'].tolist()
    recs = [movie for movie in popularity_ranking.index if movie not in already_rated]
    return recs[:k]
