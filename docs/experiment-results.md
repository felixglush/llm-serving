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

## Concurrency sweeps

Stage 9 runs the same workload sequentially at several concurrency levels. The values are sorted,
and every point keeps the model, deployment, seed, corpus version, request count, and input shape
fixed. Each point is saved as its own experiment so the existing W&B Run and Artifact mapping can
be reused:

```bash
uv run --env-file .env embedding-lab sweep \
  --service fake \
  --concurrency-values 1,2,4,8 \
  --requests 20 \
  --warmup-requests 2 \
  --results-dir results
```

The sweep manifest is written to `results/sweeps/<sweep-id>/sweep.json`; its `experiment_ids` point
to the individual experiment directories. Add `--publish` to publish one W&B Run per point.
