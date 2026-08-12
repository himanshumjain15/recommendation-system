FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 wget unzip && \
    mkdir -p sample_data/inputs && \
    wget -q -O /tmp/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip && \
    unzip -q /tmp/ml-latest-small.zip -d sample_data/inputs && \
    rm /tmp/ml-latest-small.zip && \
    apt-get purge -y --auto-remove wget unzip && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
