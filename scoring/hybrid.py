import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_similarity(movies):
    """Builds a movie x movie cosine similarity matrix from TF-IDF on genres."""
    genres_cleaned = movies['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer()
    genre_matrix = tfidf.fit_transform(genres_cleaned)
    return cosine_similarity(genre_matrix)


def get_hybrid_recommendations(
    user_id, train, movies, model, user_id_map, movie_idx_to_id,
    content_similarity, k=10, weight=0.1,
):
    """Blends content-based similarity and CF scores into one ranked recommendation list."""
    liked_movies = train[(train['userId'] == user_id) & (train['rating'] >= 4)]['movieId'].tolist()
    already_rated = train[train['userId'] == user_id]['movieId'].tolist()

    if len(liked_movies) == 0 or user_id not in user_id_map:
        return []

    liked_indices = movies[movies['movieId'].isin(liked_movies)].index.tolist()
    content_scores = content_similarity[liked_indices].mean(axis=0)
    content_scores_series = pd.Series(content_scores, index=movies['movieId'])

    user_idx = user_id_map[user_id]
    cf_scores_full = model.user_factors[user_idx] @ model.item_factors.T
    cf_scores_series = pd.Series(
        cf_scores_full, index=[movie_idx_to_id[i] for i in range(len(movie_idx_to_id))]
    )

    combined = pd.DataFrame({'content': content_scores_series, 'cf': cf_scores_series}).dropna()
    combined['content_norm'] = (
        (combined['content'] - combined['content'].min())
        / (combined['content'].max() - combined['content'].min())
    )
    combined['cf_norm'] = (
        (combined['cf'] - combined['cf'].min())
        / (combined['cf'].max() - combined['cf'].min())
    )
    combined['hybrid_score'] = weight * combined['content_norm'] + (1 - weight) * combined['cf_norm']

    combined = combined[~combined.index.isin(already_rated)]

    return combined.sort_values('hybrid_score', ascending=False).head(k).index.tolist()
