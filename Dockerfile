FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install build-essential git sqlite3 curl -y && \
    pip install .

WORKDIR /airsenal

COPY . /airsenal

CMD ["airsenal_run_pipeline"]
