FROM python:3.13

WORKDIR /pkg
COPY . /pkg

# ffmpeg might depends on a lot of image processing libraries
# that are common with the ones our Python dependencies use.
RUN apt update && apt install -y ffmpeg
RUN pip install --upgrade pip

RUN python -m venv .venv
RUN . .venv/bin/activate \
    && VISION_TOOLKIT_BUILD=py pip install .[test] \
    && VISION_TOOLKIT_BUILD=c pip install .[test]

ENV BASH_ENV="/pkg/.venv/bin/activate"
ENTRYPOINT ["/bin/bash", "-c"]
