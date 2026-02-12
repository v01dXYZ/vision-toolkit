FROM python:3.13-slim

ENV VISION_TOOLKIT_CYTHON_CACHE=1
ENV CC="ccache gcc"
ENV CXX="ccache g++"

RUN apt update && apt install -y git g++ ccache sqlite3

COPY ./setup.py ./pyproject.toml /src/
COPY ./src/ /src/src/

ARG PYPKGMGR=pip

RUN python -m venv /venv/ \
    && . /venv/bin/activate \
    && pip install --upgrade $(echo $PYPKGMGR | cut -d' ' -f1) \
    && $PYPKGMGR install /src/.[test]

WORKDIR /src

ENV BASH_ENV="/venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
