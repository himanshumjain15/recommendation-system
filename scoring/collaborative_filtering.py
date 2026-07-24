from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


def build_cf_model(train, factors=50, iterations=20, regularization=0.01, random_state=42):
    """Trains an ALS collaborative filtering model on the train set.

    Returns (model, user_id_map, movie_id_map, movie_idx_to_id, user_item_matrix).
    """
    user_ids = train['userId'].unique()
    movie_ids_train = train['movieId'].unique()

    user_id_map = {id: idx for idx, id in enumerate(user_ids)}
    movie_id_map = {id: idx for idx, id in enumerate(movie_ids_train)}
    movie_idx_to_id = {idx: id for id, idx in movie_id_map.items()}

    train = train.copy()
    train['user_idx'] = train['userId'].map(user_id_map)
    train['movie_idx'] = train['movieId'].map(movie_id_map)

    user_item_matrix = csr_matrix((train['rating'], (train['user_idx'], train['movie_idx'])))

    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, random_state=random_state,
    )
    model.fit(user_item_matrix)

    return model, user_id_map, movie_id_map, movie_idx_to_id, user_item_matrix
