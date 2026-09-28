"""Prometheus adapter for simulator telemetry."""

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class PrometheusSimulatorTelemetry:
    """Translate one simulator process's events into Prometheus metrics."""

    def __init__(
        self,
        *,
        registry: CollectorRegistry,
        configured_capacity: int,
    ) -> None:
        self._received = Counter(
            "model_server_simulator_embedding_requests_received_total",
            "Embedding requests received by this simulator process.",
            registry=registry,
        )
        self._completed = Counter(
            "model_server_simulator_embedding_requests_completed_total",
            "Embedding requests that reached a terminal simulator outcome.",
            ("outcome",),
            registry=registry,
        )
        self._rejected = Counter(
            "model_server_simulator_embedding_requests_rejected_total",
            "Embedding requests rejected before synthetic embedding work.",
            ("reason",),
            registry=registry,
        )
        self._duration = Histogram(
            "model_server_simulator_embedding_request_duration_seconds",
            "Simulator embedding request duration, including configured delay.",
            ("outcome",),
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self._active = Gauge(
            "model_server_simulator_active_requests",
            "Embedding work currently admitted by this simulator process.",
            registry=registry,
        )
        self._capacity = Gauge(
            "model_server_simulator_configured_capacity",
            "Configured maximum concurrent embedding work for this simulator process.",
            registry=registry,
        )
        self._active.set(0)
        self._capacity.set(configured_capacity)

    def request_received(self) -> None:
        self._received.inc()

    def request_completed(self, *, outcome: str, duration_seconds: float) -> None:
        self._completed.labels(outcome=outcome).inc()
        self._duration.labels(outcome=outcome).observe(duration_seconds)

    def request_rejected(self, *, reason: str) -> None:
        self._rejected.labels(reason=reason).inc()

    def work_started(self) -> None:
        self._active.inc()

    def work_completed(self) -> None:
        self._active.dec()


def install_metrics_endpoint(app: FastAPI, *, registry: CollectorRegistry) -> None:
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        from prometheus_client import generate_latest

        return Response(
            content=generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )
