FROM python:3.13

WORKDIR /pkg
COPY . /pkg

RUN pip install --upgrade pip

RUN python -m venv .venv
RUN . .venv/bin/activate \
    && VISION_TOOLKIT_BUILD=py pip install .[test] \
    && VISION_TOOLKIT_BUILD=c pip install --no-deps .[test]

ENV BASH_ENV="/pkg/.venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
