"""Run one configurable synthetic embedding backend as a standalone process."""

import argparse
from collections.abc import Sequence

import uvicorn
from prometheus_client import CollectorRegistry

from model_server_simulator.http_server import ModelServerSimulatorSettings, create_app
from model_server_simulator.prometheus_metrics import (
    PrometheusSimulatorTelemetry,
    install_metrics_endpoint,
)
from model_server_simulator.synthetic_model import SyntheticEmbeddingService


def main(argv: Sequence[str] | None = None) -> int:
    """Parse startup configuration, compose the adapter, and run one ASGI worker."""

    parser = argparse.ArgumentParser(prog="model-server-simulator")
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--model", default="lab-embedding")
    parser.add_argument("--dimensions", type=int, default=8)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--fail-every-n-requests", type=int)
    parser.add_argument("--injected-error-status", type=int, default=503)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")

    try:
        settings = ModelServerSimulatorSettings(
            backend_id=args.backend_id,
            model=args.model,
            dimensions=args.dimensions,
            delay_seconds=args.delay_seconds,
            max_concurrency=args.max_concurrency,
            fail_every_n_requests=args.fail_every_n_requests,
            injected_error_status=args.injected_error_status,
        )
    except ValueError as error:
        parser.error(str(error))

    service = SyntheticEmbeddingService(dimensions=settings.dimensions)
    registry = CollectorRegistry()
    telemetry = PrometheusSimulatorTelemetry(
        registry=registry,
        configured_capacity=settings.max_concurrency,
    )
    app = create_app(settings, service, telemetry=telemetry)
    install_metrics_endpoint(app, registry=registry)
    # Capacity and deterministic request ordinals are process-local, so this
    # learning server intentionally runs a single worker.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
