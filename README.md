# llm-serving

A monorepo of LLM serving course work. Each course lives in its own top-level folder with its own
`pyproject.toml`, `uv.lock`, docs, and tests, so courses can evolve independently.

| Folder | What it covers |
| --- | --- |
| [`embedding-serving-lab/`](embedding-serving-lab/docs/architecture.md) | Embedding serving with vLLM: benchmarks, Modal deployment, a round-robin gateway, and Prometheus/Grafana observability. |

Run commands from inside a course folder, for example:

```sh
cd embedding-serving-lab
uv sync --all-packages --all-extras
uv run pytest
```

Shared across courses: agent skills in `.agents/skills/`, Claude Code settings in `.claude/`, and
repo-wide docs such as [Runpod setup](docs/runpod.md).
