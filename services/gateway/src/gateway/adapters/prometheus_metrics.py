"""Prometheus adapter for the framework-neutral gateway telemetry port."""

from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram

from gateway.domain import BackendId

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class PrometheusGatewayTelemetry:
    """Translate gateway events into bounded Prometheus metric families."""

    def __init__(
        self,
        *,
        registry: CollectorRegistry,
        policy: str,
        backend_ids: Sequence[str],
    ) -> None:
        self._policy = policy
        self._received = Counter(
            "gateway_embedding_requests_received_total",
            "Embedding requests received by the gateway.",
            ("route",),
            registry=registry,
        )
        self._completed = Counter(
            "gateway_embedding_requests_completed_total",
            "Embedding requests that reached a terminal gateway outcome.",
            ("route", "outcome"),
            registry=registry,
        )
        self._rejected = Counter(
            "gateway_embedding_requests_rejected_total",
            "Embedding requests rejected before upstream work.",
            ("route", "reason"),
            registry=registry,
        )
        self._request_duration = Histogram(
            "gateway_embedding_request_duration_seconds",
            "Gateway embedding request duration, including request body upload.",
            ("route", "outcome"),
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self._selections = Counter(
            "gateway_backend_selections_total",
            "Backend selections made by the configured routing policy.",
            ("backend", "policy"),
            registry=registry,
        )
        self._upstream_completed = Counter(
            "gateway_upstream_requests_completed_total",
            "Selected upstream requests that reached a terminal outcome.",
            ("backend", "outcome"),
            registry=registry,
        )
        self._upstream_duration = Histogram(
            "gateway_upstream_request_duration_seconds",
            "Selected upstream duration including transport, waiting, and response validation.",
            ("backend", "outcome"),
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self._outstanding = Gauge(
            "gateway_upstream_requests_outstanding",
            "Selected upstream requests that have not finished.",
            ("backend",),
            registry=registry,
        )
        for backend_id in backend_ids:
            self._outstanding.labels(backend=backend_id).set(0)

    def request_received(self, *, route: str) -> None:
        self._received.labels(route=route).inc()

    def request_completed(self, *, route: str, outcome: str, duration_seconds: float) -> None:
        self._completed.labels(route=route, outcome=outcome).inc()
        self._request_duration.labels(route=route, outcome=outcome).observe(duration_seconds)

    def request_rejected(self, *, route: str, reason: str) -> None:
        self._rejected.labels(route=route, reason=reason).inc()

    def backend_selected(self, *, backend_id: BackendId) -> None:
        self._selections.labels(backend=backend_id, policy=self._policy).inc()

    def upstream_started(self, *, backend_id: BackendId) -> None:
        self._outstanding.labels(backend=backend_id).inc()

    def upstream_completed(
        self,
        *,
        backend_id: BackendId,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        self._outstanding.labels(backend=backend_id).dec()
        self._upstream_completed.labels(backend=backend_id, outcome=outcome).inc()
        self._upstream_duration.labels(backend=backend_id, outcome=outcome).observe(
            duration_seconds
        )


def install_metrics_endpoint(app: FastAPI, *, registry: CollectorRegistry) -> None:
    """Install an exact `/metrics` scrape route without instrumenting the scrape itself."""

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        from prometheus_client import generate_latest

        return Response(
            content=generate_latest(registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )
