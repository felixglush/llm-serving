"""Compose and run the standalone gateway process."""

import argparse
import logging
import math
import os
from collections.abc import Sequence

import uvicorn
from prometheus_client import CollectorRegistry

from gateway.adapters.fastapi_http import create_app
from gateway.adapters.httpx_backend import HttpxBackendClient
from gateway.adapters.prometheus_metrics import (
    PrometheusGatewayTelemetry,
    install_metrics_endpoint,
)
from gateway.application import Gateway
from gateway.domain import BackendId
from gateway.routing import BackendPool, RoundRobinSelection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gateway")
    parser.add_argument(
        "--backend",
        action="append",
        required=True,
        metavar="ID=BASE_URL",
        help="repeat for each configured embedding backend",
    )
    parser.add_argument(
        "--backend-api-key-env",
        action="append",
        default=[],
        metavar="ID=ENV_VAR",
        help="read one backend credential from the named environment variable",
    )
    parser.add_argument("--model", default="lab-embedding")
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if args.dimensions < 1:
        parser.error("--dimensions must be positive")
    if not math.isfinite(args.request_timeout_seconds) or args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be positive and finite")

    try:
        backend_urls = _parse_assignments(args.backend, option="--backend")
        key_environments = _parse_assignments(
            args.backend_api_key_env,
            option="--backend-api-key-env",
        )
    except ValueError as error:
        parser.error(str(error))
    unknown_key_ids = set(key_environments) - set(backend_urls)
    if unknown_key_ids:
        parser.error("--backend-api-key-env must reference a configured backend ID")

    clients = []
    for backend_id, base_url in backend_urls.items():
        api_key = None
        if backend_id in key_environments:
            environment_name = key_environments[backend_id]
            api_key = os.environ.get(environment_name)
            if not api_key:
                parser.error(f"environment variable {environment_name!r} must not be empty")
        clients.append(
            HttpxBackendClient(
                backend_id=BackendId(backend_id),
                base_url=base_url,
                api_key=api_key,
            )
        )

    registry = CollectorRegistry()
    telemetry = PrometheusGatewayTelemetry(
        registry=registry,
        policy="round_robin",
        backend_ids=tuple(backend_urls),
    )
    pool = BackendPool(clients, RoundRobinSelection())
    gateway = Gateway(
        model=args.model,
        dimensions=args.dimensions,
        pool=pool,
        telemetry=telemetry,
    )
    request_logger = logging.getLogger("gateway.requests")
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False
    if not request_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        request_logger.addHandler(handler)
    app = create_app(
        gateway=gateway,
        request_timeout_seconds=args.request_timeout_seconds,
        telemetry=telemetry,
    )
    install_metrics_endpoint(app, registry=registry)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


def _parse_assignments(values: Sequence[str], *, option: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        key, separator, configured_value = value.partition("=")
        if not separator or not key.strip() or not configured_value.strip():
            raise ValueError(f"{option} must use ID=VALUE")
        key = key.strip()
        if key in assignments:
            raise ValueError(f"{option} backend IDs must be unique")
        assignments[key] = configured_value.strip()
    return assignments


if __name__ == "__main__":
    raise SystemExit(main())
