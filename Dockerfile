FROM python:3.14-slim-bookworm

WORKDIR /airsenal
COPY . /airsenal

RUN apt-get update && \
    apt-get install build-essential git sqlite3 curl -y && \
    pip install --upgrade pip && \
    pip install .[dev]

CMD ["airsenal", "run"]
