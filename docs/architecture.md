# Architecture boundaries

The lab follows a simple dependency direction:

```text
entrypoints -> application -> ports <- adapters
                  |
                  v
                domain
```

The domain describes concepts such as a model, a replica topology, a workload, and a benchmark
result. It is pure Python and has no imports from Modal, vLLM, `httpx`, or any cloud SDK.

Application use cases coordinate those concepts through small protocols in `ports/`. Adapters
implement the protocols for a particular boundary: an in-process deterministic test adapter or an
OpenAI-compatible vLLM HTTP client. Entry points translate CLI arguments into application calls.

The vLLM adapter also owns engine-specific scheduler settings. `ReplicaTopology` describes the
portable shape of a deployment, while `VllmEngineConfig` turns the topology for one replica into
vLLM command-line flags. See [the local vLLM smoke path](local-vllm-smoke.md) for the manual
end-to-end boundary check.

The important test is architectural: importing `embedding_lab.domain` must remain possible in an
environment that has none of the serving or cloud dependencies installed. Provider configuration
belongs in an adapter configuration object; domain objects use `tensor_parallel_size` to describe
one replica spread across workers rather than naming a vendor GPU string.

Each materially different serving configuration gets a deterministic deployment identity. The
identity includes the model revision, topology, vLLM scheduler settings, GPU type, image/runtime
versions, and container concurrency. Benchmark workloads do not affect deployment identity, so a
concurrency or input-length sweep reuses the same endpoint instead of silently changing the server.

## Deployable services

Deployable processes live under `services/`, with their package manifest, container definition,
operating guide, source, and tests colocated. The existing `embedding_lab` package remains the
provider-portable research client. Gateway code lives in the sibling `services/gateway/` package,
not in the lab's model-deployment adapters.

```text
                         HTTP
gateway            ------------------+
embedding_lab      ----------------+  |
                                     v  v
                            /v1/embeddings
                           contract version 1
                              ^           ^
                              |           |
                 model-server-simulator      model-server
                 synthetic process    real model process
```

There are no shared Python runtime packages between these components. Each process owns its domain
objects, application ports, implementation, and dependencies. The shared artifact is the serialized
HTTP promise in [`contracts/embeddings-v1.openapi.json`](../contracts/embeddings-v1.openapi.json),
not FastAPI code or importable dataclasses. Clients translate that wire representation into their
own application types.

### Standalone model server simulator

The model server simulator is a separate process, not part of the Modal or vLLM deployment adapter.
Its composition root injects `SyntheticEmbeddingService` through its local `EmbeddingService` port
into the FastAPI adapter:

```text
model-server-simulator entry point
          |
          +--> SyntheticEmbeddingService --> EmbeddingService port
          |
          +--> FastAPI HTTP adapter --> /v1/embeddings, /health
```

This keeps framework and process configuration at the edge. The HTTP adapter knows the port, but
does not import the synthetic implementation, Modal, vLLM, or deployment code. The composition root
is the only place that selects the deterministic simulator. The service package owns its HTTP runtime
dependencies and can run as its own local process or container. These types intentionally resemble
some research-lab types but are not imported from the lab: independent ownership is more valuable
here than sharing a few small dataclasses.
See the [model server simulator](../services/model-server-simulator/README.md).

### Real model server

`services/model-server/` owns the real vLLM process configuration, NVIDIA Dockerfile, Modal
deployment adapter, deployment CLI, and their tests. `embedding_lab.adapters.vllm` retains only the
HTTP client and response parser because those belong to the experiment side of the boundary.

The Docker and Modal paths both launch the same pinned `vllm serve` API. They are alternative
deployment mechanisms, not two logical services. See the
[model server](../services/model-server/README.md).

### Gateway

`services/gateway/` owns the inbound HTTP adapter, embedding-routing use case, backend pool,
selection strategies, outbound HTTP adapter, process configuration, and container. It does not
import Python code from the research lab or either model server.

```text
FastAPI adapter -> Gateway -> BackendPool -> BackendClient <- HTTPX adapter
                                  |
                                  +-> RoundRobinSelection
```

The application depends on gateway-owned values and narrow interfaces. FastAPI and HTTPX stay in
outward adapters. The composition root selects concrete adapters and the round-robin strategy.
`BackendPool` keeps selection and reservation in one short critical section, but never holds its
lock during network I/O. This prepares the strategy seam for least-outstanding selection without
exposing reservation lifecycle to HTTP handlers.

### Local observability

The gateway and each simulator expose Prometheus metrics through their own HTTP adapter. The
application code depends on a small telemetry protocol. It does not import the Prometheus client.
The process composition root selects the Prometheus adapter and its process-local registry.

```text
client -> gateway -> simulator A
                  -> simulator B

gateway /metrics -----+
simulator A /metrics --+--> Prometheus <-- Grafana
simulator B /metrics --+
```

Prometheus pulls metrics through the Docker network. Grafana queries Prometheus. Request IDs stay
in JSON logs and do not become metric labels. Simulator identity is a Prometheus target label. The
simulator does not repeat deployment identity on every metric. See the
[local observability stack](../deploy/observability/README.md).
