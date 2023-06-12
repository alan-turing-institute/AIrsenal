FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get install build-essential git sqlite3 curl -y && \
    pip install -U setuptools pygmo poetry

WORKDIR /airsenal

COPY . /airsenal

RUN poetry install --extras "api"

CMD ["poetry", "run", "airsenal_run_pipeline"]
