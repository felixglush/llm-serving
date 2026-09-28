"""Integration tests for the gateway's outbound HTTP adapter."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from gateway.adapters.fastapi_http import create_app
from gateway.adapters.httpx_backend import HttpxBackendClient
from gateway.application import Gateway
from gateway.domain import BackendId, EmbedCommand, RequestContext
from gateway.failures import InvalidUpstreamResponse, UpstreamUnavailable
from gateway.routing import BackendPool, RoundRobinSelection


def _command() -> EmbedCommand:
    return EmbedCommand(
        model="lab-embedding",
        inputs=("alpha", "beta"),
        encoding_format="float",
    )


def _context() -> RequestContext:
    loop = asyncio.get_running_loop()
    return RequestContext(request_id="gateway-request-id", deadline=loop.time() + 1)


def _valid_payload() -> dict[str, object]:
    return {
        "object": "list",
        "model": "lab-embedding",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }


async def test_backend_uses_only_its_configured_credential_and_gateway_request_id() -> None:
    captured_request: httpx.Request | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=_valid_payload())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    backend = HttpxBackendClient(
        backend_id=BackendId("backend-a"),
        base_url="https://backend.example/v1",
        api_key="backend-secret",
        client=http_client,
    )
    try:
        response = await backend.embed(_command(), _context())
    finally:
        await backend.aclose()

    assert response.model == "lab-embedding"
    assert captured_request is not None
    assert captured_request.url == "https://backend.example/v1/embeddings"
    assert captured_request.headers["Authorization"] == "Bearer backend-secret"
    assert captured_request.headers["X-Request-ID"] == "gateway-request-id"
    assert "caller-secret" not in str(captured_request.headers)


async def test_public_caller_headers_cannot_cross_the_outbound_http_seam() -> None:
    captured_request: httpx.Request | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=_valid_payload())

    backend = HttpxBackendClient(
        backend_id=BackendId("backend-a"),
        base_url="https://backend.example",
        api_key="backend-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    gateway = Gateway(
        model="lab-embedding",
        dimensions=2,
        pool=BackendPool([backend], RoundRobinSelection()),
    )
    app = create_app(gateway=gateway, request_timeout_seconds=1)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={
                    "model": "lab-embedding",
                    "input": ["alpha", "beta"],
                    "encoding_format": "float",
                },
                headers={
                    "Authorization": "Bearer caller-secret",
                    "X-Caller-Only": "do-not-forward",
                },
            )
    finally:
        await gateway.aclose()

    assert response.status_code == 200
    assert captured_request is not None
    assert captured_request.headers["Authorization"] == "Bearer backend-secret"
    assert "X-Caller-Only" not in captured_request.headers
    assert "caller-secret" not in str(captured_request.headers)


async def test_each_backend_receives_only_its_own_configured_credential() -> None:
    credentials_seen: dict[str, str] = {}

    def responder(backend_id: str) -> Callable[[httpx.Request], httpx.Response]:
        def respond(request: httpx.Request) -> httpx.Response:
            credentials_seen[backend_id] = request.headers["Authorization"]
            return httpx.Response(200, json=_valid_payload())

        return respond

    backends = [
        HttpxBackendClient(
            backend_id=BackendId(backend_id),
            base_url=f"https://{backend_id}.example",
            api_key=secret,
            client=httpx.AsyncClient(transport=httpx.MockTransport(responder(backend_id))),
        )
        for backend_id, secret in (("backend-a", "secret-a"), ("backend-b", "secret-b"))
    ]
    gateway = Gateway(
        model="lab-embedding",
        dimensions=2,
        pool=BackendPool(backends, RoundRobinSelection()),
    )
    app = create_app(gateway=gateway, request_timeout_seconds=1)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
        ) as client:
            for _ in range(2):
                response = await client.post(
                    "/v1/embeddings",
                    json={
                        "model": "lab-embedding",
                        "input": ["alpha", "beta"],
                        "encoding_format": "float",
                    },
                )
                assert response.status_code == 200
    finally:
        await gateway.aclose()

    assert credentials_seen == {
        "backend-a": "Bearer secret-a",
        "backend-b": "Bearer secret-b",
    }


async def test_readiness_probe_uses_the_selected_backends_configured_credential() -> None:
    authorization: str | None = None

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal authorization
        authorization = request.headers.get("Authorization")
        return httpx.Response(200)

    backend = HttpxBackendClient(
        backend_id=BackendId("backend-a"),
        base_url="https://backend.example/v1",
        api_key="backend-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        ready = await backend.is_ready()
    finally:
        await backend.aclose()

    assert ready is True
    assert authorization == "Bearer backend-secret"


async def test_backend_normalizes_valid_embeddings_by_their_indices() -> None:
    payload = _valid_payload()
    data = payload["data"]
    assert isinstance(data, list)
    payload["data"] = list(reversed(data))
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    backend = HttpxBackendClient(
        backend_id=BackendId("backend-a"),
        base_url="http://backend",
        client=http_client,
    )
    try:
        response = await backend.embed(_command(), _context())
    finally:
        await backend.aclose()

    assert [item.index for item in response.embeddings] == [0, 1]
    assert response.embeddings[0].values == (0.1, 0.2)
    assert response.embeddings[1].values == (0.3, 0.4)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"model": "lab-embedding"}),
        httpx.Response(
            200,
            json={
                **_valid_payload(),
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                    {"object": "embedding", "index": 0, "embedding": [0.2]},
                ],
            },
        ),
        httpx.Response(
            200,
            json={
                **_valid_payload(),
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [10**400, 0.2],
                    },
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
            },
        ),
    ],
)
async def test_malformed_backend_payload_is_a_typed_failure(response: httpx.Response) -> None:
    backend = HttpxBackendClient(
        backend_id=BackendId("backend-a"),
        base_url="http://backend",
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)),
    )
    try:
        with pytest.raises(InvalidUpstreamResponse):
            await backend.embed(_command(), _context())
    finally:
        await backend.aclose()


async def test_backend_http_failure_is_unavailable_without_retry() -> None:
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "secret upstream detail"}})

    backend = HttpxBackendClient(
        backend_id=BackendId("backend-a"),
        base_url="http://backend",
        client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        with pytest.raises(UpstreamUnavailable) as captured:
            await backend.embed(_command(), _context())
    finally:
        await backend.aclose()

    assert calls == 1
    assert "secret upstream detail" not in str(captured.value)
