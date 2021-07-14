FROM python:3.8-slim-buster

ARG APP_DIR=/app
WORKDIR ${APP_DIR}
COPY . ${APP_DIR}
RUN chmod -R a+w ${APP_DIR}

RUN apt update
RUN apt install -y --no-install-recommends git
RUN rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
RUN pip install -r requirements.txt
