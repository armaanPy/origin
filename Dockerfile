FROM python:3.12-slim AS dev

WORKDIR /origin

RUN pip install uv==0.6.9
RUN apt-get update && apt-get install -y git
COPY pyproject.toml uv.lock ./
COPY src ./

RUN uv venv /origin/.venv && \
    . /origin/.venv/bin/activate && \
    uv pip install -e .

ENV PATH="/origin/.venv/bin:$PATH"

ENV PYTHONPATH="/origin:$PYTHONPATH"
