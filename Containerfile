FROM python:3.13-slim

RUN apt update && apt install -y git g++
RUN pip install --upgrade pip

WORKDIR /pkg
COPY . /pkg

RUN python -m venv .venv \
    && . .venv/bin/activate \
    && VISION_TOOLKIT_BUILD=py pip install .[test] \
    && VISION_TOOLKIT_BUILD=c pip install --no-deps .[test]

ENV BASH_ENV="/pkg/.venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
