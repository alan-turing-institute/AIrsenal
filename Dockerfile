FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install build-essential git sqlite3 curl -y && \
    pip install -U uv

WORKDIR /airsenal

COPY . /airsenal

RUN uv sync --extra api

CMD ["uv", "run", "airsenal_run_pipeline"]
