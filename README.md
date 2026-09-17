# Embedding Serving Lab

This repository is a small, reproducible laboratory for embedding inference. The initial target
is Qwen3-Embedding-0.6B served through vLLM's OpenAI-compatible `/v1/embeddings` API on one Modal
L4 container. The application code keeps that choice at the edge so the same benchmark can later
exercise another serving engine or deployment platform.

The project grows as a stack of functional slices:

1. vendor-neutral domain and project foundation
2. embedding-service port and deterministic fake
3. service verification
4. vLLM HTTP adapter
5. vLLM runtime configuration
6. Modal deployment adapter
7. closed-loop benchmark
8. experiment result artifacts
9. concurrency sweep

The first slices run offline. A real vLLM or Modal check is an integration exercise and requires
the corresponding service URL and credentials at runtime; secrets are never written to result
artifacts.

## Development

```bash
UV_CACHE_DIR=/private/tmp/llm-serving-uv-cache uv sync --extra dev --extra modal
uv run ruff check .
uv run mypy
uv run pytest
```

`uv.lock` records the exact development environment after dependency resolution. The default
domain package does not import Modal, vLLM, or an HTTP client.
