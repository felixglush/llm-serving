import httpx
import pytest

from embedding_lab.adapters.gateway.http_client import HttpxGatewayClient
from embedding_lab.ports.gateway_client import (
    GatewayHttpError,
    GatewayProtocolError,
    GatewayRequestTimeout,
    GatewayTransportError,
)


def response_payload(*, count: int = 1) -> dict[str, object]:
    return {
        "object": "list",
        "model": "lab-embedding",
        "data": [
            {"object": "embedding", "index": index, "embedding": [float(index), 1.0]}
            for index in range(count)
        ],
        "usage": {"prompt_tokens": count, "total_tokens": count},
    }


async def test_http_client_returns_validated_success_and_gateway_backend_identity() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://gateway.test/v1/embeddings"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            headers={"X-Backend-ID": "backend-b"},
            json=response_payload(count=2),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport_client:
        async with HttpxGatewayClient(
            base_url="https://gateway.test",
            model="lab-embedding",
            api_key="secret",
            timeout_seconds=2.0,
            client=transport_client,
        ) as client:
            result = await client.embed(["alpha", "beta"])

    assert result.backend_id == "backend-b"
    assert result.vectors == 2
    assert result.input_tokens == 2


async def test_http_client_preserves_bounded_http_error_classification() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "upstream_unavailable",
                    "type": "upstream_unavailable",
                    "message": "backend failed",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport_client:
        client = HttpxGatewayClient(
            base_url="https://gateway.test/v1",
            model="lab-embedding",
            timeout_seconds=2.0,
            client=transport_client,
        )
        with pytest.raises(GatewayHttpError) as raised:
            await client.embed(["alpha"])

    assert raised.value.status_code == 503
    assert raised.value.error_code == "upstream_unavailable"
    assert str(raised.value) == "backend failed"


async def test_http_client_preserves_redirect_as_a_non_successful_http_outcome() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"Location": "https://other.test/v1/embeddings"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport_client:
        client = HttpxGatewayClient(
            base_url="https://gateway.test",
            model="lab-embedding",
            timeout_seconds=2.0,
            client=transport_client,
        )
        with pytest.raises(GatewayHttpError) as raised:
            await client.embed(["alpha"])

    assert raised.value.status_code == 307
    assert raised.value.error_code is None


@pytest.mark.parametrize(
    ("handler_error", "expected"),
    [
        (httpx.ReadTimeout("slow"), GatewayRequestTimeout),
        (httpx.ConnectError("offline"), GatewayTransportError),
    ],
)
async def test_http_client_distinguishes_timeout_from_other_transport_failures(
    handler_error: Exception,
    expected: type[Exception],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(handler_error, httpx.RequestError):
            handler_error.request = request
        raise handler_error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport_client:
        client = HttpxGatewayClient(
            base_url="https://gateway.test",
            model="lab-embedding",
            timeout_seconds=2.0,
            client=transport_client,
        )
        with pytest.raises(expected):
            await client.embed(["alpha"])


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, headers={"X-Backend-ID": "backend-a"}, content=b"not-json"),
        httpx.Response(200, json=response_payload()),
        httpx.Response(200, headers={"X-Backend-ID": "backend-a"}, json={"data": []}),
        httpx.Response(
            200,
            headers={"X-Backend-ID": "backend-a"},
            json={**response_payload(), "object": "embedding"},
        ),
        httpx.Response(
            200,
            headers={"X-Backend-ID": "backend-a"},
            json={**response_payload(), "model": "different-model"},
        ),
        httpx.Response(
            200,
            headers={"X-Backend-ID": "backend-a"},
            json={
                **response_payload(),
                "data": [{"object": "not-embedding", "index": 0, "embedding": [1.0]}],
            },
        ),
        httpx.Response(
            200,
            headers={"X-Backend-ID": "backend-a", "Content-Type": "application/json"},
            content=(
                b'{"object":"list","model":"lab-embedding","data":'
                b'[{"object":"embedding","index":0,"embedding":[NaN]}],'
                b'"usage":{"prompt_tokens":1,"total_tokens":1}}'
            ),
        ),
    ],
)
async def test_http_client_rejects_malformed_success_responses(response: httpx.Response) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport_client:
        client = HttpxGatewayClient(
            base_url="https://gateway.test",
            model="lab-embedding",
            timeout_seconds=2.0,
            client=transport_client,
        )
        with pytest.raises(GatewayProtocolError):
            await client.embed(["alpha"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_url": "not-a-url", "model": "lab-embedding"}, "HTTP or HTTPS"),
        ({"base_url": "https://gateway.test", "model": " "}, "model"),
        (
            {"base_url": "https://gateway.test", "model": "lab-embedding", "timeout_seconds": 0},
            "timeout_seconds",
        ),
    ],
)
def test_http_client_configuration_fails_before_network_work(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HttpxGatewayClient(**kwargs)  # type: ignore[arg-type]
