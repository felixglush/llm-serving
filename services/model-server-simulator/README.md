# Model server simulator

The model server simulator is a small standalone microservice for embedding-gateway development.
It owns its deterministic synthetic embedding implementation. It does not load a model, deploy
vLLM, or call Modal. Token counts are synthetic (one per input string) and must not be presented as
measured model throughput. Its current API scope is embeddings. The broader service name leaves
room for other explicitly implemented model behaviors later.

## Install the server dependencies

```bash
UV_CACHE_DIR=/private/tmp/llm-serving-uv-cache uv sync --all-extras --all-packages
```

The `model-server-simulator` workspace package owns its FastAPI and Uvicorn runtime dependencies.
FastAPI translates the HTTP contract into Python values. Uvicorn is the ASGI server: roughly, the
process that listens on a TCP port and calls the FastAPI application for each request. The service
does not import another repository package or install the research lab, Modal deployment code,
vLLM, or W&B.

## Run backends A and B

Open two terminals from the repository root.

Terminal A:

```bash
uv run --package model-server-simulator model-server-simulator \
  --backend-id simulator-a \
  --port 8101 \
  --dimensions 8 \
  --delay-seconds 0.05 \
  --max-concurrency 2
```

Terminal B:

```bash
uv run --package model-server-simulator model-server-simulator \
  --backend-id simulator-b \
  --port 8102 \
  --dimensions 8 \
  --delay-seconds 0.15 \
  --max-concurrency 2
```

Both processes use the same deterministic vector function, so identical input and dimensions
produce identical vectors. Backend identity changes headers and health output, not embeddings.

Check health:

```bash
curl --fail http://127.0.0.1:8101/health
curl --fail http://127.0.0.1:8102/health
```

Send an embedding request:

```bash
curl --fail-with-body http://127.0.0.1:8101/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"lab-embedding","input":["first text","second text"],"encoding_format":"float"}'
```

The supported request shape is deliberately smaller than the OpenAI API. `model` must match the
server configuration, `input` must be a nonempty list of strings, and `encoding_format` must be
`float`. The response contains ordered, indexed vectors and synthetic usage fields. The
`X-Backend-ID` response header identifies the serving backend.

The simulator has fixed safety ceilings of 1 MiB per request body, 256 input strings, and 8,192
characters per string. These protect the standalone test backend before admission. They are not
claimed as model capacity; the later gateway admission layer will provide operator-configurable
workload limits.

## Capacity, delay, and repeatable failures

`--max-concurrency` limits accepted embedding work in this one process. When every slot is in use,
the server immediately returns `429` with `Retry-After: 1`; it does not queue the request. The slot
is released after success, failure, or cancellation. `--delay-seconds` uses asynchronous sleep, so
health checks remain responsive while embedding work is delayed.

To fail every second accepted request with a repeatable `503`:

```bash
uv run --package model-server-simulator model-server-simulator \
  --backend-id faulty \
  --port 8103 \
  --fail-every-n-requests 2 \
  --injected-error-status 503
```

Invalid requests and capacity rejections do not advance the accepted-request counter. There is no
public fault-control endpoint; fault behavior is fixed at process startup.

## Run as its own container

The dedicated image installs only the service package and its third-party HTTP runtime. It does not
install another repository package, the Modal or W&B extras, or the pre-existing deployment CLI.

```bash
docker build -t model-server-simulator services/model-server-simulator
docker run --rm -p 127.0.0.1:8101:8000 model-server-simulator \
  --backend-id simulator-a --host 0.0.0.0 --port 8000
```

Run a second container on another host port (for example `-p 127.0.0.1:8102:8000`) with a
different backend ID. The explicit `127.0.0.1` publishes this unauthenticated learning service only
to the laptop, while `--host 0.0.0.0` lets Uvicorn accept traffic inside the container. Each
container intentionally runs one Uvicorn worker because capacity and deterministic fault counters
are process-local.
