FROM python:3.8-slim-buster

WORKDIR /airsenal

COPY . /airsenal

RUN apt-get update && apt-get install build-essential -y && \
    pip install .

CMD ["airsenal_run_pipeline"]
