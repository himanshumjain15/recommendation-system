def build_held_out_liked(held_out, rating_threshold=4):
    """Maps each user to the list of movieIds they rated >= threshold in the held-out set."""
    return held_out[held_out['rating'] >= rating_threshold].groupby('userId')['movieId'].apply(list).to_dict()


def get_hits(recommendations, held_out_liked, k=10):
    hits = 0
    total_users = 0

    for user, liked_movies in held_out_liked.items():
        if user not in recommendations:
            continue
        total_users += 1
        recs = recommendations[user][:k]
        if any(movie in recs for movie in liked_movies):
            hits += 1

    return hits / total_users


def get_precision_recall(recommendations, held_out_liked, k=10):
    precisions = []
    recalls = []

    for user, liked_movies in held_out_liked.items():
        if user not in recommendations:
            continue
        recs = recommendations[user][:k]
        overlap = len(set(recs) & set(liked_movies))
        precisions.append(overlap / k)
        recalls.append(overlap / len(liked_movies))

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)
