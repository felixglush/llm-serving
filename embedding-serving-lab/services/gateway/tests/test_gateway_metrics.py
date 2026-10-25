"""Observable gateway metrics through the public scrape endpoint."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import pytest
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from gateway.adapters.fastapi_http import create_app
from gateway.adapters.prometheus_metrics import (
    PrometheusGatewayTelemetry,
    install_metrics_endpoint,
)
from gateway.application import Gateway
from gateway.domain import BackendId, EmbedCommand, Embedding, EmbeddingResponse, RequestContext
from gateway.failures import UpstreamUnavailable
from gateway.routing import BackendPool, RoundRobinSelection


def _success(*, dimensions: int = 3) -> EmbeddingResponse:
    return EmbeddingResponse(
        model="lab-embedding",
        embeddings=(Embedding(index=0, values=tuple(float(i) for i in range(dimensions))),),
        prompt_tokens=1,
        total_tokens=1,
    )


@dataclass(slots=True)
class ScriptedBackend:
    backend_id: BackendId
    outcomes: list[EmbeddingResponse | Exception]
    delay_seconds: float = 0.0
    calls: int = 0
    active: int = 0

    async def embed(self, command: EmbedCommand, context: RequestContext) -> EmbeddingResponse:
        del command, context
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        self.active += 1
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            self.active -= 1

    async def is_ready(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def _metrics_client(
    backends: Sequence[ScriptedBackend],
    *,
    timeout_seconds: float = 1.0,
) -> AsyncIterator[httpx.AsyncClient]:
    registry = CollectorRegistry()
    telemetry = PrometheusGatewayTelemetry(
        registry=registry,
        policy="round_robin",
        backend_ids=tuple(str(backend.backend_id) for backend in backends),
    )
    pool = BackendPool(backends, RoundRobinSelection())
    gateway = Gateway(
        model="lab-embedding",
        dimensions=3,
        pool=pool,
        telemetry=telemetry,
    )
    app = create_app(
        gateway=gateway,
        request_timeout_seconds=timeout_seconds,
        telemetry=telemetry,
    )
    install_metrics_endpoint(app, registry=registry)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        yield client
    await gateway.aclose()


def _request_body(*, model: str = "lab-embedding") -> dict[str, object]:
    return {"model": model, "input": ["alpha"], "encoding_format": "float"}


async def _scrape(client: httpx.AsyncClient) -> str:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return response.text


def _sample_value(
    exposition: str,
    name: str,
    expected_labels: dict[str, str] | None = None,
) -> float:
    expected_labels = expected_labels or {}
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(key) == value for key, value in expected_labels.items()
            ):
                return float(sample.value)
    return 0.0


async def test_metrics_describe_success_and_validation_rejection_without_payload_labels() -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async with _metrics_client([backend]) as client:
        rejected = await client.post(
            "/v1/embeddings",
            json=_request_body(model="unknown-model"),
        )
        succeeded = await client.post("/v1/embeddings", json=_request_body())
        exposition = await _scrape(client)

    assert rejected.status_code == 400
    assert succeeded.status_code == 200
    route = {"route": "/v1/embeddings"}
    assert _sample_value(exposition, "gateway_embedding_requests_received_total", route) == 2
    assert (
        _sample_value(
            exposition,
            "gateway_embedding_requests_completed_total",
            route | {"outcome": "succeeded"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_embedding_requests_completed_total",
            route | {"outcome": "unknown_model"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_embedding_requests_rejected_total",
            route | {"reason": "unknown_model"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_backend_selections_total",
            {"backend": "backend-a", "policy": "round_robin"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_upstream_requests_completed_total",
            {"backend": "backend-a", "outcome": "succeeded"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_upstream_requests_outstanding",
            {"backend": "backend-a"},
        )
        == 0
    )
    assert (
        _sample_value(
            exposition,
            "gateway_embedding_request_duration_seconds_count",
            route | {"outcome": "succeeded"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_upstream_request_duration_seconds_count",
            {"backend": "backend-a", "outcome": "succeeded"},
        )
        == 1
    )

    forbidden = {"request_id", "input", "model", "client_id", "url", "error_message"}
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            if sample.name.startswith("gateway_"):
                assert forbidden.isdisjoint(sample.labels)


@pytest.mark.parametrize(
    ("outcome", "backend_result", "expected_status"),
    [
        ("upstream_unavailable", UpstreamUnavailable(), 503),
        ("invalid_upstream_response", _success(dimensions=2), 502),
    ],
)
async def test_upstream_failures_are_counted_once_and_release_the_outstanding_gauge(
    outcome: str,
    backend_result: EmbeddingResponse | Exception,
    expected_status: int,
) -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [backend_result])

    async with _metrics_client([backend]) as client:
        response = await client.post("/v1/embeddings", json=_request_body())
        exposition = await _scrape(client)

    assert response.status_code == expected_status
    assert (
        _sample_value(
            exposition,
            "gateway_upstream_requests_completed_total",
            {"backend": "backend-a", "outcome": outcome},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "gateway_upstream_requests_outstanding",
            {"backend": "backend-a"},
        )
        == 0
    )


async def test_deadline_and_cancellation_are_observable_without_leaking_outstanding_work() -> None:
    backend = ScriptedBackend(
        BackendId("backend-a"),
        [_success(), _success()],
        delay_seconds=0.2,
    )

    async with _metrics_client([backend], timeout_seconds=0.03) as client:
        timed_out = await client.post("/v1/embeddings", json=_request_body())
        backend.delay_seconds = 0.2
        pending = asyncio.create_task(client.post("/v1/embeddings", json=_request_body()))
        async with asyncio.timeout(1):
            while backend.active == 0:
                await asyncio.sleep(0)
        during = await _scrape(client)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        after = await _scrape(client)

    assert timed_out.status_code == 504
    assert (
        _sample_value(
            during,
            "gateway_upstream_requests_outstanding",
            {"backend": "backend-a"},
        )
        == 1
    )
    assert (
        _sample_value(
            after,
            "gateway_upstream_requests_outstanding",
            {"backend": "backend-a"},
        )
        == 0
    )
    assert (
        _sample_value(
            after,
            "gateway_upstream_requests_completed_total",
            {"backend": "backend-a", "outcome": "deadline_exceeded"},
        )
        == 1
    )
    assert (
        _sample_value(
            after,
            "gateway_upstream_requests_completed_total",
            {"backend": "backend-a", "outcome": "cancelled"},
        )
        == 1
    )


async def test_health_readiness_and_scrapes_are_not_embedding_workload() -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async with _metrics_client([backend]) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 200
        first = await _scrape(client)
        second = await _scrape(client)

    assert _sample_value(first, "gateway_embedding_requests_received_total") == 0
    assert _sample_value(second, "gateway_embedding_requests_received_total") == 0
