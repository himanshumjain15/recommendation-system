CREATE TABLE users (
    user_id INTEGER PRIMARY KEY
);

CREATE TABLE items (
    movie_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT
);

CREATE TABLE interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL REFERENCES items(movie_id),
    rating NUMERIC(2, 1) NOT NULL,
    rated_at TIMESTAMP NOT NULL
);

CREATE TABLE recommendation_logs (
    log_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL REFERENCES items(movie_id),
    model_used TEXT NOT NULL,
    served_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE experiment_assignments (
    user_id INTEGER PRIMARY KEY REFERENCES users(user_id),
    group_name TEXT NOT NULL,
    assigned_at TIMESTAMP NOT NULL DEFAULT NOW()
);
