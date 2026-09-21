---
name: benchmark-modal-vllm-embeddings
description: Benchmark a vLLM OpenAI-compatible embedding endpoint deployed on Modal, using vLLM's reference benchmark and optionally a portable project benchmark. Apply when validating throughput, latency, concurrency, or reproducibility for `/v1/embeddings`; do not use for in-process `LLM.embed()` benchmarks or generative-model token benchmarks.
---

# Benchmark Modal vLLM Embeddings

Measure a deployed embedding service without contaminating the GPU server or confusing network latency with engine capacity. Preserve enough context and raw output to compare later runs.

## Establish the measurement boundary

Before running load, identify and record:

- endpoint URL and whether it is Modal proxy authenticated;
- model ID and pinned revision;
- vLLM version and relevant engine flags;
- GPU type, replicas, and tensor-parallel size;
- benchmark client location;
- dataset, input-length policy, request count, concurrency or request rate, and seed.

Use two client placements for different questions:

- A separate CPU client in the same Modal routing region measures service capacity with little WAN noise.
- A client in the expected user region measures end-to-end latency.

Never run the canonical load generator inside the GPU serving container. It competes with vLLM for resources and bypasses the normal Modal ingress path. If a localhost run is useful for diagnosis, label it explicitly and do not compare it directly with routed results.

## Check current interfaces

vLLM and Modal change quickly. Before constructing commands:

1. Run `vllm bench serve --help` in the benchmark-client environment.
2. Check the current vLLM embedding benchmark documentation when flags or result fields are uncertain: <https://docs.vllm.ai/en/latest/benchmarking/cli/>.
3. Check current Modal Server behavior when authentication, routing, scaling, or cold starts matter: <https://modal.com/docs/guide/servers>.

For sequence embeddings, use the OpenAI-compatible path:

```text
--backend openai-embeddings
--endpoint /v1/embeddings
```

Do not use chat/completion backends or generation metrics such as TTFT and output-token throughput for this workload.

## Prepare the server

Benchmark an existing deployment unless the user also requested deployment. Do not silently create or scale paid GPU resources.

Confirm that the deployed topology matches the experiment. A single-replica study must not autoscale horizontally during the run. Record any warm or replacement containers that could affect routing.

Modal Servers can return `503` while scaling from zero. Poll `/health` with a bounded deadline until it returns `200`, then perform explicit warmup requests. Keep cold-start latency out of steady-state results; measure it as a separate experiment when requested. Stop if readiness never succeeds rather than retrying indefinitely.

Run the repository's correctness check before load when it is available. Correct vector count, dimensions, finite values, and reported input-token usage are prerequisites for a meaningful performance result.

## Handle authentication

Modal proxy tokens may be sent as one bearer value composed from the token ID and secret. Pass credentials from environment variables and keep their values out of source files, result JSON, metadata, shell traces, logs, PRs, and summaries.

Use a literal environment-variable reference in commands, for example:

```text
--header "Authorization=Bearer $MODAL_PROXY_TOKEN"
```

Do not configure a second vLLM API key on the same bearer header when Modal proxy authentication already protects the endpoint. If both layers must authenticate, use separate Modal proxy headers and verify the benchmark client supports them.

## Run the reference benchmark

Construct the command from the installed CLI's help. A typical text-embedding run has this shape:

```bash
vllm bench serve \
  --backend openai-embeddings \
  --base-url "$EMBEDDING_BASE_URL" \
  --endpoint /v1/embeddings \
  --model "$EMBEDDING_MODEL" \
  --dataset-name random \
  --num-prompts 500 \
  --max-concurrency 8 \
  --header "Authorization=Bearer $MODAL_PROXY_TOKEN" \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_DIR"
```

Treat this as a template, not a promise that every flag is unchanged. Prefer fixed seeds and explicit input-length settings when supported. Do not mix a concurrency cap with an arrival-rate claim without recording both; the achieved request rate may be lower when concurrency is saturated.

Start with a small calibration run. For a concurrency sweep, vary only concurrency, commonly `1, 2, 4, 8, 16, 32`, while holding the server, inputs, request count, client location, and warmup policy fixed. Run points sequentially so simultaneous clients do not contaminate one another.

## Use the portable benchmark as the canonical experiment

When the repository provides a provider-neutral benchmark runner, execute it against the same endpoint and workload after the vLLM reference run. The vLLM benchmark is an engine-maintained calibration tool; the portable runner is the canonical source for cross-engine and cross-platform comparisons.

Compare overlapping facts such as:

- successful and failed requests;
- request-latency percentiles;
- vectors per second;
- reported input tokens per second;
- effective concurrency or request rate.

Investigate material disagreement before accepting results. Common causes include different input distributions, batching semantics, warmups, retries, client locations, or token-count definitions.

## Preserve evidence

Keep the untouched vLLM result file and, when requested, its detailed per-request output. Store normalized project results separately rather than rewriting the raw reference output.

Every comparison should retain:

- UTC timestamp and experiment ID;
- model and revision;
- vLLM version;
- deployment topology and engine configuration;
- client location and benchmark command with secret values redacted;
- workload definition and warmup policy;
- raw result path and normalized result path;
- failures, timeouts, and any observed Modal `429`, `503`, or preemption events.

Do not claim two runs are comparable when one of these material dimensions differs. Report those differences as limitations.

## Completion criteria

A benchmark workflow is complete when correctness passed, the endpoint was warm, the load generator ran outside the GPU server, raw results were preserved without credentials, configuration and client location were recorded, and the conclusions distinguish service capacity from end-to-end network latency.
