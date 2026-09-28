FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

COPY requirements.lock pyproject.toml ./

RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python --require-hashes -r requirements.lock

COPY src src

RUN uv pip install --python /app/.venv/bin/python --no-deps .

USER 65532:65532
EXPOSE 8000

ENTRYPOINT ["/app/.venv/bin/gateway"]
