# vLLM model server

This service owns the real OpenAI-compatible embedding runtime and its deployment configuration.
It is intentionally separate from `model-simulator`, which returns deterministic synthetic vectors.

There are two equivalent runtime paths:

- `Dockerfile` extends vLLM's official NVIDIA image and runs `vllm serve` directly.
- `vllm_model_server.modal_deployment` builds a Modal-managed GPU image, then launches the same
  `vllm serve` command behind a Modal web-server endpoint.

The research package remains a client of the resulting HTTP endpoint. Benchmarking, result storage,
and W&B publication are not owned by this service. The service owns its Python configuration types;
its cross-service promise is the versioned
[`/v1/embeddings` contract](../../contracts/embeddings-v1.openapi.json), not a shared Python package.

## NVIDIA Docker runtime

Build on a Linux NVIDIA host (or CI builder with the required target architecture). The image is
pinned to vLLM 0.30.0, the current release used by both Docker and Modal paths:

```bash
docker build -t vllm-model-server services/vllm-model-server
```

Run with explicit GPU access and persistent caches:

```bash
docker run --rm --gpus all --ipc=host \
  --user 2000:0 \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/home/vllm/.cache/huggingface" \
  -v vllm-cache:/home/vllm/.cache/vllm \
  --env HF_TOKEN \
  vllm-model-server \
  Qwen/Qwen3-Embedding-0.6B --runner pooling --host 0.0.0.0 --port 8000
```

`--ipc=host` gives PyTorch enough shared memory for multiprocessing and tensor parallelism. The
container is NVIDIA/Linux-specific; an Apple Silicon build is not production-runtime evidence.

## Modal deployment

Install the control-plane dependency and inspect the command:

```bash
uv sync --all-packages --extra modal
uv run --package vllm-model-server vllm-model-server-deploy --help
```

The deploy command creates cloud resources and is therefore run only with explicit approval. Modal
constructs its own GPU image from the pinned vLLM version; it does not run this repository's Docker
image. Both paths are owned here because they expose the same vLLM server contract.
