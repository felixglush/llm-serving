# Modal embedding deployment

This runbook creates one Modal deployment for one complete serving configuration. Different models,
model revisions, GPU types, or runtime settings receive different deterministic app names. Benchmark
workloads reuse an existing deployment and therefore do not appear in the deployment fingerprint.

## Install and authenticate

Install the optional Modal adapter dependency:

```bash
uv sync --extra modal
uv run modal --version
uv run modal token info
```

Creating or updating a deployment allocates external cloud resources. Review the generated code and
configuration before running the deployment command.

## Deploy the initial configuration

Use an immutable model revision when recording reproducible experiments. `main` is accepted, but it
can move and therefore does not uniquely identify downloaded weights.

```bash
uv run embedding-lab deploy \
  --provider modal \
  --model Qwen/Qwen3-Embedding-0.6B \
  --revision main \
  --max-sequence-length 8192 \
  --embedding-dimensions 1024 \
  --replicas 1 \
  --tensor-parallel-size 1 \
  --gpu-type L4 \
  --container-concurrency 32 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192
```

The command prints JSON containing the deterministic app name, configuration fingerprint, and Modal
endpoint URL. Stage 6 enforces one replica on one worker and `max_containers=1`; multi-replica and
tensor-parallel deployments remain later stages.

Modal may scale the deployment to zero because `min_containers` defaults to zero. The first request
can therefore include model startup time. Warm the endpoint before measuring steady-state latency.

## Verify the real endpoint

```bash
export EMBEDDING_BASE_URL="https://...modal.run"

uv run embedding-lab verify \
  --service vllm \
  --model Qwen/Qwen3-Embedding-0.6B \
  --expected-dimensions 1024 \
  --timeout-seconds 900 \
  --text "The capital of Canada is Ottawa." \
  --text "Gravity attracts objects with mass."
```

The Modal adapter mounts separate shared volumes for Hugging Face and vLLM caches. Sharing these
caches across deployments avoids repeated downloads for steady-state experiments. A future cold-start
experiment must record or isolate cache state because a warm cache changes startup behavior.
