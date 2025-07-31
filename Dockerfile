FROM python:3.12-slim-bookworm

WORKDIR /airsenal
COPY . /airsenal

RUN apt-get update && \
    apt-get install build-essential git sqlite3 curl -y && \
    pip install .

CMD ["airsenal_run_pipeline"]
