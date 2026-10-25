# Local observability stack

This Compose topology runs two model server simulators, the gateway, Prometheus, and Grafana on one
private Docker network. Containers use service names for internal connections. `localhost` inside a
container refers only to that container.

Start the stack from the repository root:

```bash
docker compose --file deploy/observability/compose.yaml up --build --wait
```

The host endpoints are bound to the loopback interface:

- gateway: <http://127.0.0.1:8000>
- simulator A: <http://127.0.0.1:8101>
- simulator B: <http://127.0.0.1:8102>
- Prometheus: <http://127.0.0.1:9090>
- Grafana: <http://127.0.0.1:3000> (`admin` / `admin` by default)

Grafana loads the Prometheus data source and the **Embedding Gateway Traffic** dashboard from files.
Backend B has a two-second synthetic delay so that its higher duration and outstanding work are easy
to see. Histograms estimate percentiles. Gateway upstream duration includes HTTP transport, backend
waiting, and response validation; it is not model-only processing time.

The Compose file also accepts `BACKEND_A_DELAY_SECONDS`, `BACKEND_B_DELAY_SECONDS`,
`BACKEND_A_MAX_CONCURRENCY`, `BACKEND_B_MAX_CONCURRENCY`, and
`GATEWAY_REQUEST_TIMEOUT_SECONDS`. These variables change the synthetic topology at process
startup. They do not change gateway routing policy. The fixed-arrival experiment recipes set these
values through reviewed `.env` files under `experiments/gateway-load/`.

Stop the stack and remove its temporary metrics database:

```bash
docker compose --file deploy/observability/compose.yaml down --volumes
```

Run the real acceptance test with:

```bash
RUN_OBSERVABILITY_E2E=1 .venv/bin/pytest -q tests/test_observability_stack.py
```

The Stage 4 load acceptance test uses the same real containers and network with bounded synthetic
traffic:

```bash
RUN_GATEWAY_LOAD_E2E=1 .venv/bin/pytest -q tests/test_gateway_load_e2e.py
```
