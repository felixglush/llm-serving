# Experiment results

Stage 8 saves each benchmark before publishing it:

```text
results/<experiment-id>/
  experiment.json
  observations.json
  wandb-publication.json  # after a successful W&B publication
```

Install the tracking dependency and configure the project:

```bash
uv sync --extra wandb
cp .env.example .env
# Put WANDB_API_KEY in .env, or use `wandb login` once.
```

Run a small offline experiment:

```bash
uv run --env-file .env embedding-lab experiment \
  --service fake \
  --requests 10 \
  --concurrency 2 \
  --warmup-requests 1
```

Publish the saved record to the configured project by adding `--publish`. The publisher uses the
experiment ID as the W&B run ID with `resume="allow"`, groups runs by deployment fingerprint, logs
comparison metrics once, and attaches the local JSON files as one `benchmark` Artifact. A local
publication receipt makes repeated publication of the same experiment a no-op.

For a live service, provide its endpoint and immutable model revision. Warm-up is recorded before
the measured clock; it is not included in steady-state latency or throughput metrics. W&B SDK system
metrics belong to the machine running the benchmark client and must not be interpreted as Modal GPU
utilization.
