FROM python:3.9-bullseye

COPY event_warping /build/event_warping
COPY event_warping_extension /build/event_warping_extension
COPY setup.py /build/setup.py
COPY scripts /build/scripts
COPY example.py /build/example.py
COPY README.md /build/README.md

WORKDIR /build
RUN python3 -m pip install -e .
