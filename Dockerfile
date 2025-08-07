FROM python:3.12-slim-bookworm

WORKDIR /airsenal
COPY . /airsenal

RUN apt-get update && \
    apt-get install build-essential git sqlite3 curl -y && \
    pip install --upgrade pip && \
    pip install .[dev,api]

CMD ["airsenal_run_pipeline"]
