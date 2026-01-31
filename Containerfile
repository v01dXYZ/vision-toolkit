FROM python:3.13-slim AS builder

RUN apt update && apt install -y git g++

WORKDIR /pkg
COPY . /pkg

ARG PYPKGMGR=pip

RUN python -m venv /venv \
    && . /venv/bin/activate \
    && pip install --upgrade $(echo $PYPKGMGR | cut -d' ' -f1) \
    && VISION_TOOLKIT_BUILD=py $PYPKGMGR install .[test] \
    && VISION_TOOLKIT_BUILD=c $PYPKGMGR install --no-deps .[test]

FROM python:3.13-slim

COPY --from=builder /venv /venv

RUN apt update && apt install -y git

WORKDIR /src

ENV BASH_ENV="/venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
# 3