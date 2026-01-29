FROM python:3.13

WORKDIR /pkg
COPY . /pkg

RUN pip install --upgrade pip

RUN python -m venv .venv
RUN . .venv/bin/activate && pip install .[test]

ENV BASH_ENV="/pkg/.venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
