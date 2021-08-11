FROM python:3.8-slim-buster

WORKDIR /airsenal

COPY . /airsenal

RUN apt-get update && apt-get install build-essential git sqlite3 -y && \
    pip install pygmo && pip install .

CMD ["airsenal_run_pipeline"]
