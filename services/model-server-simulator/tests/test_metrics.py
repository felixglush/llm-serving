"""Observable simulator metrics through its public scrape endpoint."""

import asyncio

import httpx
import pytest
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from model_server_simulator.http_server import ModelServerSimulatorSettings, create_app
from model_server_simulator.prometheus_metrics import (
    PrometheusSimulatorTelemetry,
    install_metrics_endpoint,
)
from model_server_simulator.synthetic_model import SyntheticEmbeddingService


def _settings(
    *,
    delay_seconds: float = 0.0,
    max_concurrency: int = 1,
    fail_every_n_requests: int | None = None,
) -> ModelServerSimulatorSettings:
    return ModelServerSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=delay_seconds,
        max_concurrency=max_concurrency,
        fail_every_n_requests=fail_every_n_requests,
        injected_error_status=503,
    )


def _client(
    settings: ModelServerSimulatorSettings,
) -> tuple[httpx.AsyncClient, PrometheusSimulatorTelemetry]:
    registry = CollectorRegistry()
    telemetry = PrometheusSimulatorTelemetry(
        registry=registry,
        configured_capacity=settings.max_concurrency,
    )
    app = create_app(
        settings,
        SyntheticEmbeddingService(dimensions=settings.dimensions),
        telemetry=telemetry,
    )
    install_metrics_endpoint(app, registry=registry)
    return (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://simulator",
        ),
        telemetry,
    )


def _body() -> dict[str, object]:
    return {"model": "lab-embedding", "input": ["alpha"], "encoding_format": "float"}


async def _scrape(client: httpx.AsyncClient) -> str:
    response = await client.get("/metrics")
    assert response.status_code == 200
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


async def test_metrics_count_success_validation_rejection_and_injected_failure() -> None:
    settings = _settings(fail_every_n_requests=2)
    client, _ = _client(settings)

    async with client:
        invalid = await client.post(
            "/v1/embeddings",
            json={"model": "wrong", "input": ["alpha"], "encoding_format": "float"},
        )
        succeeded = await client.post("/v1/embeddings", json=_body())
        injected = await client.post("/v1/embeddings", json=_body())
        exposition = await _scrape(client)

    assert invalid.status_code == 400
    assert succeeded.status_code == 200
    assert injected.status_code == 503
    assert (
        _sample_value(
            exposition,
            "model_server_simulator_embedding_requests_received_total",
        )
        == 3
    )
    assert (
        _sample_value(
            exposition,
            "model_server_simulator_embedding_requests_completed_total",
            {"outcome": "invalid_request"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "model_server_simulator_embedding_requests_completed_total",
            {"outcome": "succeeded"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "model_server_simulator_embedding_requests_completed_total",
            {"outcome": "injected_error"},
        )
        == 1
    )
    assert (
        _sample_value(
            exposition,
            "model_server_simulator_embedding_requests_rejected_total",
            {"reason": "invalid_request"},
        )
        == 1
    )
    assert _sample_value(exposition, "model_server_simulator_configured_capacity") == 1
    assert _sample_value(exposition, "model_server_simulator_active_requests") == 0

    forbidden = {"backend", "request_id", "input", "model", "client_id", "url"}
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            if sample.name.startswith("model_server_simulator_"):
                assert forbidden.isdisjoint(sample.labels)


async def test_active_work_rises_during_a_request_and_returns_to_zero_after_success() -> None:
    settings = _settings(delay_seconds=0.15)
    client, _ = _client(settings)

    async with client:
        request = asyncio.create_task(client.post("/v1/embeddings", json=_body()))
        async with asyncio.timeout(1):
            while True:
                during = await _scrape(client)
                if _sample_value(during, "model_server_simulator_active_requests") == 1:
                    break
                await asyncio.sleep(0)
        response = await request
        after = await _scrape(client)

    assert response.status_code == 200
    assert _sample_value(after, "model_server_simulator_active_requests") == 0


async def test_capacity_rejection_and_cancellation_are_counted_without_leaking_work() -> None:
    settings = _settings(delay_seconds=0.2)
    client, _ = _client(settings)

    async with client:
        pending = asyncio.create_task(client.post("/v1/embeddings", json=_body()))
        async with asyncio.timeout(1):
            while True:
                during = await _scrape(client)
                if _sample_value(during, "model_server_simulator_active_requests") == 1:
                    break
                await asyncio.sleep(0)

        rejected = await client.post("/v1/embeddings", json=_body())
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        after = await _scrape(client)

    assert rejected.status_code == 429
    assert (
        _sample_value(
            after,
            "model_server_simulator_embedding_requests_rejected_total",
            {"reason": "capacity_exhausted"},
        )
        == 1
    )
    assert (
        _sample_value(
            after,
            "model_server_simulator_embedding_requests_completed_total",
            {"outcome": "cancelled"},
        )
        == 1
    )
    assert _sample_value(after, "model_server_simulator_active_requests") == 0


async def test_health_and_scrapes_do_not_count_as_embedding_workload() -> None:
    client, _ = _client(_settings())

    async with client:
        assert (await client.get("/health")).status_code == 200
        first = await _scrape(client)
        second = await _scrape(client)

    assert (
        _sample_value(
            first,
            "model_server_simulator_embedding_requests_received_total",
        )
        == 0
    )
    assert (
        _sample_value(
            second,
            "model_server_simulator_embedding_requests_received_total",
        )
        == 0
    )
