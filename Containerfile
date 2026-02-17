FROM python:3.13-slim

ENV VISION_TOOLKIT_CYTHON_CACHE=1
ENV CC="ccache gcc"
ENV CXX="ccache g++"

ENV XDG_CACHE_HOME=/cache
ENV CYTHON_CACHE_DIR=/cache/cython
ENV CCACHE_DIR=/cache/ccache
ENV CCACHE_NOHASHDIR=true

RUN apt update && apt install -y git g++ ccache sqlite3

COPY ./setup.py ./pyproject.toml /src/
COPY ./src/ /src/src/

ARG PYPKGMGR=pip
ARG PYPKGMGR_INSTALL_OPTS="--no-build-isolation"

RUN python -m venv /venv/ \
    && . /venv/bin/activate \
    && pip install --upgrade $(echo $PYPKGMGR | cut -d' ' -f1) \
    && cd /src \
    && $PYPKGMGR install setuptools numpy Cython pybind11 \
    && $PYPKGMGR install $PYPKGMGR_INSTALL_OPTS .[test] \
    && rm -r build
# normally $PYPKGMGR_INSTALL_OPTS contains --no-build-isolation
# no-build-isolation is important to let ccache not have different include directories
# without build isolation, ccache needs to run preprocessor (it is not a lot slower though)

WORKDIR /src

ENV BASH_ENV="/venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
