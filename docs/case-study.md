# CineMatch: a hybrid movie recommender, tested and deployed

Case study content for the portfolio site. Sections match the `ProjectCaseStudy` format.

**Live demo**: https://himanshumjain15-recsys.streamlit.app
**Code**: https://github.com/himanshumjain15/recommendation-system


## Overview

A movie recommendation system that learns what someone will enjoy from how people behave,
not from what films are labelled as. It runs as a live web service on AWS with a public
demo, and its central claim was tested with a proper A/B experiment rather than asserted.

The result: personalized recommendations found something a viewer genuinely liked for 43
out of every 100 viewers. Showing everyone the same popular films worked for 31. That gap
is statistically significant at p = 0.00072.

Most recommender projects stop at a notebook with an accuracy score. This one ships: a
Postgres database, a REST API, containerized deployment, an experiment with a
pre-registered success threshold, and a front end anyone can click.


## Results

From the A/B experiment, with viewers split into two groups and each group served by a
different model:

| Group | Hit-rate@10 | n | 95% CI |
|---|---|---|---|
| Control (popularity) | 29.35% | 293 | 24.43% to 34.81% |
| Treatment (hybrid) | 42.54% | 315 | 37.20% to 48.06% |

**Lift: 13.19 percentage points, a 44.9% relative improvement, p = 0.00072.**

The lift's confidence interval runs from 5.64 to 20.74 points. Even its pessimistic end
clears the 4.67-point threshold set before the experiment ran, so the result is practically
as well as statistically significant.

Hit-rate@10 counts a viewer as a hit if at least one of the ten recommended films was
something they went on to rate highly in data the model never saw.


## Problem statement and approach

Every team building recommendations faces the same question before investing in
personalization: is it actually worth it? A "most popular" list is nearly free to build
and surprisingly hard to beat, because popular films really are widely liked.

So the project was framed as a hypothesis to test, not a model to build:

> **H0**: a personalized hybrid model performs about the same as a popularity baseline.
> **H1**: the hybrid performs significantly better.

Three models were built to answer it:

| Approach | How it decides | Hit-rate@10 |
|---|---|---|
| Popularity | Most-rated films, same for everyone | 31.13% |
| Content-based | Films with matching genres | 5.25% |
| Collaborative filtering | What similar-behaving people liked | 42.13% |
| **Hybrid** (90% CF, 10% content) | **Both signals blended** | **43.65%** |

Two findings from that table were not obvious in advance.

**Content-based filtering, alone, was the worst approach by a wide margin.** It scores 5 in
100, six times worse than simply showing popular films. It also produces the most
convincing-looking output: ask it for recommendations for an action fan and you get ten
action films. It looks right and it doesn't work, because genre tags say nothing about
whether a film is any good or whether anyone will actually watch it.

**Exploratory analysis drove the design rather than following it.** The interaction matrix
was 98.3% empty, which is why collaborative filtering alone was expected to struggle for
sparse users and why a content signal was worth blending in at all. Separately, the top 10%
of films absorbed 60% of all ratings, which is what made the popularity baseline a serious
opponent rather than a strawman.


## System architecture

MovieLens data feeds Postgres, which trains three models. FastAPI serves them and logs
every recommendation back to Postgres. The Streamlit demo calls that API over HTTP.

- **Data layer**: Postgres with five tables (`users`, `items`, `interactions`,
  `recommendation_logs`, `experiment_assignments`), schema initialized automatically on
  first container start.
- **Models**: ALS matrix factorization via `implicit` for collaborative filtering, TF-IDF
  genre vectors with cosine similarity for content, blended by a tuned weight.
- **Serving**: FastAPI loads the trained model and similarity matrix once at startup, then
  answers requests in milliseconds. Every recommendation served is written to Postgres.
- **Experimentation**: users bucketed deterministically into control and treatment,
  each group served by a different model, results compared with a two-proportion test.
- **Front end**: a Streamlit app that calls the live API rather than holding its own copy
  of the model, so what a visitor sees is the deployed system's real output.


## Key features

**Three models, compared honestly.** The demo shows the same viewer's recommendations from
all three approaches side by side, each labelled with its measured hit rate. The weakest
one is included precisely because it looks the most plausible.

**Recommendations that log themselves.** Every response written to
`recommendation_logs` with the model that produced it. Without that audit trail the A/B
analysis would have been guesswork rather than measurement.

**Deterministic bucketing.** Users are assigned to experiment groups with an MD5 hash of
their ID rather than a random draw, so assignments are reproducible and auditable. Python's
built-in `hash()` was rejected for this: on integers it returns the number itself, so
`hash(user_id) % 2` would have quietly split users by odd and even.

**Pre-registered success criteria.** The minimum detectable effect (15% relative lift) and
the analysis method were fixed before looking at any results, so the threshold could not be
rationalized after the fact.

**Personas grounded in real data.** The demo's example viewers are real MovieLens users
selected for genuinely dominant taste, each shown alongside their actual genre breakdown.
An earlier version labelled a user "the comedy lover" who turned out to be 57% drama and
only 36% comedy; the labels were corrected once the breakdown was displayed next to them.


## Technical stack

**Modeling**: Python, pandas, NumPy, scikit-learn (TF-IDF, cosine similarity), `implicit`
(ALS), SciPy and statsmodels for the significance test

**Backend**: FastAPI, PostgreSQL, SQLAlchemy, psycopg2

**Infrastructure**: Docker, Docker Compose, AWS EC2, Streamlit Community Cloud

**Front end**: Streamlit with custom CSS, TMDB API for poster art


## Deployment

Two pieces, deployed separately.

The API and database run on a single AWS EC2 instance using the same Docker Compose file as
local development, so there is no separate deployment configuration that can drift. An
Elastic IP keeps the address stable across instance restarts, which matters because a
published link should not break when the server is stopped to save money.

The demo runs on Streamlit Community Cloud, deployed straight from the GitHub repository
and redeployed automatically on every push. It holds no model of its own and calls the EC2
API over HTTP.

Because the raw dataset is deliberately not committed to the repository, both the container
build and the demo download it on first run, so a fresh clone works without manual setup.


## Challenges and solutions

Six real failures, each diagnosed rather than worked around.

**A missing system library.** The compiled ALS code depends on `libgomp`, which the slim
Python base image doesn't ship. The container built cleanly and crashed on startup. This is
exactly the class of hidden dependency that containerization exists to expose: it had been
invisible locally because the host already had it.

**"localhost" meaning different things.** The API tried to reach Postgres at `localhost`,
which inside a container means the container itself, not the database next to it. Fixed by
addressing Postgres by its Compose service name.

**Out of memory on a small instance.** The content similarity matrix needs roughly 700MB;
the instance had 1GB total, shared with the OS and the database. Adding swap space solved
it without upgrading to a paid instance size.

**A silent half-outage.** After a reboot the site went down, but only partly: Postgres came
back because it had a restart policy and the API did not, because it didn't. The instance
looked healthy while serving nothing. Both services now restart automatically.

**A disk that filled up.** Repeated image builds exhausted an 8GB volume mid-deployment.
Pruning unused layers unblocked it; resizing the volume and extending the filesystem fixed
it properly.

**A cached failure that outlived its cause.** The demo first deployed with an invalid API
key, so every poster lookup returned nothing, and the cache stored those empty results.
Correcting the key changed nothing, because the cache answered instead of retrying. Only a
restart cleared it. Caching a failure is not the same as caching a result, and the app now
says so out loud when the key is missing instead of failing quietly.


## Honest limitations

Stated because they affect how much the result proves.

**This is an offline experiment, not a live one.** It measures recommendations against
what viewers historically went on to rate highly, not against how they would react to
being shown these specific films. That is counterfactual evaluation, and it is a known
limitation of offline recommender testing rather than a shortcut taken here.

**The sample is small and fixed.** With roughly 300 viewers per group, the test has about
90% power for an effect the size actually observed but only about 25% power for the
smaller effect originally set as the threshold. It reliably detects large differences and
would likely miss modest ones, and no more data can be collected.

**Tuning and testing used the same held-out data.** The blend weight was chosen by its
performance on the data later used for the significance test, which gives the hybrid a
small home advantage. A stricter design would separate a validation set for tuning from an
untouched test set.


## Improvements

**Richer content signal.** Genre tags are coarse: many unrelated films share identical tag
sets, which is a large part of why content-based scoring performed so poorly. Plot
summaries or sentence-transformer embeddings would likely change that conclusion, and the
5% result should be read as a verdict on genre tags rather than on content-based methods.

**A separate validation set** for hyperparameter tuning, removing the double-dipping caveat
above.

**Scheduled retraining.** The model is trained once and loaded at startup, so new users,
new films and new ratings are invisible until it is rebuilt. Production systems decouple
serving from retraining on a schedule.

**Cold-start handling.** Viewers with no history get nothing today. Content-based scoring
could cover them, which is the case for keeping it in the blend despite its standalone
performance.

**Approximate nearest-neighbour retrieval.** At this scale a full similarity matrix is
fine. At a realistic catalogue size it would not be, and a FAISS index would be the
standard answer.
