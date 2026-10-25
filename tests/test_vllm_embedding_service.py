import json

import httpx
import pytest

from embedding_lab.adapters.vllm.embedding_service import (
    VllmEmbeddingService,
    VllmRequestError,
)
from embedding_lab.adapters.vllm.schemas import VllmResponseError, parse_embedding_response
from embedding_lab.domain.embedding import Embedding
from embedding_lab.ports.embedding_service import EmbeddingService


async def test_vllm_adapter_sends_openai_request_and_restores_input_order() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://embeddings.example/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert json.loads(request.content) == {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": ["first", "second"],
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [3, 4.5]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 2.0]},
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    service: EmbeddingService = VllmEmbeddingService(
        base_url="https://embeddings.example/",
        model="Qwen/Qwen3-Embedding-0.6B",
        api_key="test-token",
        client=client,
    )

    response = await service.embed(["first", "second"])

    assert response.embeddings == (Embedding((1.0, 2.0)), Embedding((3.0, 4.5)))
    assert response.input_tokens == 5
    await client.aclose()


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"data": [], "usage": {}}, "prompt_tokens"),
        (
            {
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0]},
                    {"object": "embedding", "index": 0, "embedding": [2.0]},
                ],
                "usage": {"prompt_tokens": 2},
            },
            "duplicate index",
        ),
        (
            {
                "data": [{"object": "embedding", "index": 1, "embedding": [1.0]}],
                "usage": {"prompt_tokens": 1},
            },
            "indices must be 0 through 0",
        ),
        (
            {
                "data": [{"object": "embedding", "index": 0, "embedding": ["not-a-number"]}],
                "usage": {"prompt_tokens": 1},
            },
            "only numbers",
        ),
    ],
)
def test_vllm_response_requires_usage_and_one_numeric_vector_per_input(
    payload: object, message: str
) -> None:
    with pytest.raises(VllmResponseError, match=message):
        parse_embedding_response(payload, expected_count=1)


async def test_vllm_adapter_reports_http_status_without_response_body() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text="internal secret", request=request)
        )
    )
    service = VllmEmbeddingService(
        base_url="https://embeddings.example/v1",
        model="served-model",
        client=client,
    )

    with pytest.raises(VllmRequestError, match="HTTP 503") as error:
        await service.embed(["hello"])

    assert "internal secret" not in str(error.value)
    await client.aclose()


async def test_vllm_adapter_includes_bounded_structured_error_message() -> None:
    long_message = "served model does not exist " + ("x" * 600)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400,
                json={
                    "error": {
                        "message": long_message,
                        "type": "BadRequestError",
                        "code": 400,
                    }
                },
                request=request,
            )
        )
    )
    service = VllmEmbeddingService(
        base_url="https://embeddings.example/v1",
        model="wrong-model",
        client=client,
    )

    with pytest.raises(VllmRequestError) as error:
        await service.embed(["hello"])

    assert str(error.value).startswith(
        "vLLM embedding request failed with HTTP 400: served model does not exist"
    )
    assert len(str(error.value).split(": ", maxsplit=1)[1]) == 500
    await client.aclose()


async def test_vllm_adapter_reports_invalid_json() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not json"))
    )
    service = VllmEmbeddingService(
        base_url="https://embeddings.example",
        model="served-model",
        client=client,
    )

    with pytest.raises(VllmRequestError, match="invalid JSON"):
        await service.embed(["hello"])

    await client.aclose()


@pytest.mark.parametrize(
    "base_url, timeout_seconds",
    [
        ("not-a-url", 30.0),
        ("https://embeddings.example", 0.0),
        ("https://embeddings.example", float("nan")),
        ("https://embeddings.example", True),
    ],
)
def test_vllm_adapter_rejects_invalid_connection_settings(
    base_url: str, timeout_seconds: float
) -> None:
    with pytest.raises(ValueError):
        VllmEmbeddingService(
            base_url=base_url,
            model="served-model",
            timeout_seconds=timeout_seconds,
        )
