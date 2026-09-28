"""Behavioral integration tests through the gateway's public HTTP interface."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import pytest

from gateway.adapters.fastapi_http import create_app
from gateway.application import Gateway
from gateway.domain import (
    BackendId,
    EmbedCommand,
    Embedding,
    EmbeddingResponse,
    RequestContext,
)
from gateway.failures import UpstreamUnavailable
from gateway.routing import (
    BackendPool,
    BackendSelectionStrategy,
    BackendSnapshot,
    RoundRobinSelection,
)


def _success(*, dimensions: int = 3) -> EmbeddingResponse:
    return EmbeddingResponse(
        model="lab-embedding",
        embeddings=(Embedding(index=0, values=tuple(float(i) for i in range(dimensions))),),
        prompt_tokens=1,
        total_tokens=1,
    )


@dataclass(slots=True)
class ScriptedBackend:
    """Controllable adapter used only at the gateway's outbound seam."""

    backend_id: BackendId
    outcomes: list[EmbeddingResponse | Exception]
    delay_seconds: float = 0.0
    calls: int = 0
    active: int = 0
    closed: bool = False
    ready: bool = True

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

    async def aclose(self) -> None:
        self.closed = True

    async def is_ready(self) -> bool:
        return self.ready


@dataclass(slots=True)
class RecordingSelection:
    snapshots: list[tuple[BackendSnapshot, ...]] = field(default_factory=list)

    def select(self, candidates: Sequence[BackendSnapshot]) -> BackendId:
        snapshot = tuple(candidates)
        self.snapshots.append(snapshot)
        return snapshot[0].backend_id


@asynccontextmanager
async def _gateway_client(
    backends: Sequence[ScriptedBackend],
    *,
    timeout_seconds: float = 1.0,
    selection: BackendSelectionStrategy | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    pool = BackendPool(backends, selection or RoundRobinSelection())
    gateway = Gateway(model="lab-embedding", dimensions=3, pool=pool)
    app = create_app(gateway=gateway, request_timeout_seconds=timeout_seconds)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        yield client
    await gateway.aclose()


def _request_body(*, model: str = "lab-embedding") -> dict[str, object]:
    return {
        "model": model,
        "input": ["alpha"],
        "encoding_format": "float",
    }


async def test_unknown_model_is_rejected_without_calling_a_backend() -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async with _gateway_client([backend]) as client:
        response = await client.post(
            "/v1/embeddings",
            json=_request_body(model="unknown-model"),
        )

    assert response.status_code == 400
    assert response.headers["X-Request-ID"]
    assert response.json() == {
        "error": {
            "message": "model must match the configured model",
            "type": "unknown_model",
            "code": "unknown_model",
        }
    }
    assert backend.calls == 0


async def test_failed_selected_backend_is_not_retried_on_the_other_backend() -> None:
    unavailable = ScriptedBackend(
        BackendId("backend-a"),
        [UpstreamUnavailable()],
    )
    healthy = ScriptedBackend(BackendId("backend-b"), [_success()])

    async with _gateway_client([unavailable, healthy]) as client:
        response = await client.post("/v1/embeddings", json=_request_body())

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "upstream_unavailable"
    assert unavailable.calls == 1
    assert healthy.calls == 0


async def test_malformed_backend_result_is_a_bounded_bad_gateway_response() -> None:
    wrong_dimensions = ScriptedBackend(
        BackendId("backend-a"),
        [_success(dimensions=2)],
    )

    async with _gateway_client([wrong_dimensions]) as client:
        response = await client.post("/v1/embeddings", json=_request_body())

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "selected backend returned an invalid response",
            "type": "invalid_upstream_response",
            "code": "invalid_upstream_response",
        }
    }
    assert len(response.json()["error"]["message"]) <= 500


async def test_overall_deadline_returns_504_and_cleans_up_active_work() -> None:
    backend = ScriptedBackend(
        BackendId("backend-a"),
        [_success(), _success()],
        delay_seconds=0.1,
    )

    async with _gateway_client([backend], timeout_seconds=0.01) as client:
        response = await client.post("/v1/embeddings", json=_request_body())

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "deadline_exceeded"
    assert backend.calls == 1
    assert backend.active == 0


async def test_overall_deadline_includes_reading_the_request_body() -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async def slow_body() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.05)
        yield b'{"model":"lab-embedding","input":["alpha"],"encoding_format":"float"}'

    async with _gateway_client([backend], timeout_seconds=0.01) as client:
        response = await client.post(
            "/v1/embeddings",
            content=slow_body(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "deadline_exceeded"
    assert backend.calls == 0


async def test_client_cancellation_releases_the_backend_reservation() -> None:
    backend = ScriptedBackend(
        BackendId("backend-a"),
        [_success(), _success()],
        delay_seconds=0.1,
    )
    selection = RecordingSelection()

    async with _gateway_client([backend], selection=selection) as client:
        request = asyncio.create_task(client.post("/v1/embeddings", json=_request_body()))
        async with asyncio.timeout(1):
            while backend.active == 0:
                await asyncio.sleep(0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        backend.delay_seconds = 0
        later = await client.post("/v1/embeddings", json=_request_body())

    assert later.status_code == 200
    assert backend.active == 0
    assert len(selection.snapshots) == 2
    assert selection.snapshots[1][0].outstanding == 0


@pytest.mark.parametrize(
    "body",
    [
        {"model": "lab-embedding", "input": [], "encoding_format": "float"},
        {"model": "lab-embedding", "input": "alpha", "encoding_format": "float"},
        {"model": "lab-embedding", "input": ["alpha", 3], "encoding_format": "float"},
        {"model": "lab-embedding", "input": ["alpha"], "encoding_format": "base64"},
    ],
)
async def test_invalid_request_never_reaches_a_backend(body: dict[str, object]) -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async with _gateway_client([backend]) as client:
        response = await client.post("/v1/embeddings", json=body)

    assert response.status_code == 400
    assert response.headers["X-Request-ID"]
    assert set(response.json()["error"]) == {"message", "type", "code"}
    assert backend.calls == 0


@pytest.mark.parametrize(
    ("body", "expected_status"),
    [
        (
            {
                "model": "lab-embedding",
                "input": ["x"] * 257,
                "encoding_format": "float",
            },
            400,
        ),
        (
            {
                "model": "lab-embedding",
                "input": ["x" * 8_193],
                "encoding_format": "float",
            },
            400,
        ),
    ],
)
async def test_fixed_safety_bounds_reject_work_before_a_backend_call(
    body: dict[str, object], expected_status: int
) -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async with _gateway_client([backend]) as client:
        response = await client.post("/v1/embeddings", json=body)

    assert response.status_code == expected_status
    assert response.headers["X-Request-ID"]
    assert len(response.json()["error"]["message"]) <= 500
    assert backend.calls == 0


async def test_oversized_request_body_returns_413_before_a_backend_call() -> None:
    backend = ScriptedBackend(BackendId("backend-a"), [_success()])

    async with _gateway_client([backend]) as client:
        response = await client.post(
            "/v1/embeddings",
            content=b"x" * 1_048_577,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.headers["X-Request-ID"]
    assert len(response.json()["error"]["message"]) <= 500
    assert backend.calls == 0


async def test_readiness_requires_at_least_one_reachable_backend() -> None:
    unavailable = ScriptedBackend(BackendId("backend-a"), [_success()], ready=False)
    available = ScriptedBackend(BackendId("backend-b"), [_success()], ready=True)

    async with _gateway_client([unavailable, available]) as client:
        ready = await client.get("/ready")
        available.ready = False
        not_ready = await client.get("/ready")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
    assert not_ready.status_code == 503
    assert not_ready.json() == {"status": "not_ready"}
