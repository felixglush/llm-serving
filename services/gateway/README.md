# Gateway

The gateway is a standalone HTTP process that routes embedding requests across configured model
servers. Stage 2 uses round-robin selection and forwards each accepted request exactly once. It does
not retry, queue, split, or hedge requests.

The gateway, `model-server-simulator`, and `model-server` do not import Python code from each other.
They communicate through the versioned
[`/v1/embeddings` contract](../../contracts/embeddings-v1.openapi.json).

## Architecture

```text
FastAPI adapter -> Gateway -> BackendPool -> BackendClient <- HTTPX adapter
                                  |
                                  +-> RoundRobinSelection
```

`Gateway.embed()` owns the use case. `BackendPool` selects and reserves a backend atomically, then
releases its short lock before HTTP work. The reservation is released in `finally` on success,
failure, timeout, or cancellation. The selection strategy sees only configured backend IDs and
outstanding counts. It cannot access URLs, credentials, request bodies, or HTTP clients.

The FastAPI adapter owns request parsing and HTTP error mapping. The HTTPX adapter owns backend
credentials, connection reuse, safe outbound headers, and response validation. The composition root
in `main.py` is the only module that selects concrete adapters.

## Run locally

Start two model server simulators as described in
[`model-server-simulator`](../model-server-simulator/README.md), then run:

```bash
uv run --package gateway gateway \
  --backend backend-a=http://127.0.0.1:8101 \
  --backend backend-b=http://127.0.0.1:8102 \
  --model lab-embedding \
  --dimensions 8 \
  --request-timeout-seconds 2 \
  --port 8000
```

`--backend` is repeatable and keeps backend identity under operator control. To configure a backend
credential without putting it in process arguments, name its environment variable:

```bash
export BACKEND_A_API_KEY=replace-me
uv run --package gateway gateway \
  --backend backend-a=https://backend-a.example \
  --backend-api-key-env backend-a=BACKEND_A_API_KEY \
  --model lab-embedding \
  --dimensions 1024
```

The caller's `Authorization` header and arbitrary request headers are never forwarded. The HTTPX
adapter sends only the selected backend's configured credential and the gateway-generated request
ID.

Check process liveness and current backend reachability:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

## Run as its own container

The build context is the gateway directory. The image contains only the gateway package and its
HTTP runtime dependencies.

```bash
docker build -t gateway services/gateway
docker run --rm -p 127.0.0.1:8000:8000 gateway \
  --backend backend-a=http://host.docker.internal:8101 \
  --backend backend-b=http://host.docker.internal:8102 \
  --model lab-embedding \
  --dimensions 8 \
  --host 0.0.0.0 \
  --port 8000
```

`host.docker.internal` lets the container contact simulators running on the Mac host. Inside a
container, `localhost` would refer to the gateway container itself.

## Observable behavior

Successful responses include `X-Request-ID` and the configured `X-Backend-ID`. Expected failures
use bounded error envelopes: invalid requests use `400`, oversized bodies use `413`, malformed
backend responses use `502`, unavailable backends use `503`, and overall deadline expiry uses
`504`.

Completion logs are JSON on standard error. They contain request ID, configured backend ID, outcome,
and duration. They do not contain request text, vectors, URLs, or credentials.
