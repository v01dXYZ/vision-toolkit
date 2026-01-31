FROM python:3.13-slim AS builder

RUN apt update && apt install -y git g++

COPY ./setup.py ./setup.cfg ./pyproject.toml /pkg/
COPY ./src/ /pkg/src/

ARG PYPKGMGR=pip

RUN python -m venv /venv/ \
    && . /venv/bin/activate \
    && pip install --upgrade $(echo $PYPKGMGR | cut -d' ' -f1) \
    && VISION_TOOLKIT_BUILD=py $PYPKGMGR install /pkg/.[test] \
    && VISION_TOOLKIT_BUILD=c $PYPKGMGR install --no-deps /pkg/.[test]

FROM python:3.13-slim

COPY --from=builder /venv/ /venv/
COPY --from=builder /pkg/ /pkg/

RUN apt update \
    && apt install -y git

WORKDIR /src

ENV BASH_ENV="/venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
