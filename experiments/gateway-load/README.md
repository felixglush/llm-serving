# Fixed-arrival gateway experiments

This directory owns repeatable recipes for the `embedding-lab gateway-load` client. The client is
not part of the gateway service. It sends requests through the gateway's public HTTP interface and
saves its own experiment evidence under `results/gateway-load/`.

The traffic is open loop: each request has an intended arrival time derived from the configured
rate. A slow request does not delay the next planned arrival. If the client already has
`--max-outstanding` requests, it records the arrival as `generator_dropped`; it does not queue that
arrival and silently become a closed-loop test.

All three recipes use deterministic synthetic model servers. Their throughput is useful for
learning about routing and overload behavior. It is not a production model-capacity measurement.

## Common workflow

Start one topology from the repository root. `--env-file` supplies values that Compose substitutes
into the simulator and gateway process arguments:

```bash
docker compose \
  --env-file experiments/gateway-load/recipes/unequal.env \
  --file deploy/observability/compose.yaml \
  up --build --wait
```

Run a bounded experiment in another terminal:

```bash
uv run embedding-lab gateway-load \
  --base-url http://127.0.0.1:8000 \
  --model lab-embedding \
  --request-rate 20 \
  --duration-seconds 5 \
  --max-requests 100 \
  --max-outstanding 32 \
  --request-timeout-seconds 3 \
  --drain-timeout-seconds 4 \
  --texts-per-request 1 \
  --target-tokens-per-text 16 \
  --target-kind synthetic \
  --target-label local-unequal-simulators
```

The command writes these files before printing a successful result:

```text
results/gateway-load/<experiment-id>/
  gateway-load.json
  observations.json
```

`gateway-load.json` contains the complete non-secret client configuration and summary.
`observations.json` preserves
each intended dispatch time, actual dispatch time, scheduling lag, request latency, outcome, HTTP
status, and observable backend identity. The backend distribution covers responses that carry
`X-Backend-ID`; current gateway error responses do not identify a backend.

Stop the topology after the experiment:

```bash
docker compose \
  --env-file experiments/gateway-load/recipes/unequal.env \
  --file deploy/observability/compose.yaml \
  down --volumes --remove-orphans
```

## Recipe purposes

- `equal.env`: both synthetic backends have the same delay and capacity. Use it as a routing
  baseline.
- `unequal.env`: backend B is much slower than backend A. Use it to observe how capacity-blind
  round-robin routing accumulates slow unfinished work.
- `saturation.env`: both backends have one slot and a longer delay. Use a high request rate to
  produce visible upstream overload responses. Keep generator drops separate from those service
  responses.

For the saturation recipe, a useful bounded command is:

```bash
uv run embedding-lab gateway-load \
  --base-url http://127.0.0.1:8000 \
  --model lab-embedding \
  --request-rate 40 \
  --duration-seconds 2 \
  --max-requests 80 \
  --max-outstanding 16 \
  --request-timeout-seconds 2.5 \
  --drain-timeout-seconds 3 \
  --target-kind synthetic \
  --target-label local-saturation-simulators
```

Interpret the evidence in separate layers:

- intended versus actual dispatch time measures client scheduling;
- `generator_dropped` measures the configured client cap;
- HTTP status and gateway error code measure service outcomes;
- successful request latency excludes rejected and failed responses;
- arrival-window completion rate and whole-run throughput use different, explicit time windows;
- Prometheus/Grafana show service-side outstanding work and duration.

Do not subtract unrelated percentile values to infer queue time. Gateway upstream duration includes
transport, backend waiting, and response validation.
