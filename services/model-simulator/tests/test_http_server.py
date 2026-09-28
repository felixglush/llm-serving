import asyncio

import httpx
import pytest

from model_simulator.http_server import (
    MAX_INPUT_ITEMS,
    MAX_REQUEST_BYTES,
    MAX_TEXT_LENGTH,
    ModelSimulatorSettings,
    create_app,
)
from model_simulator.synthetic_model import SyntheticEmbeddingService


async def test_model_simulator_http_server_public_contract() -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.0,
        max_concurrency=2,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/embeddings",
            json={
                "model": "lab-embedding",
                "input": ["alpha", "beta"],
                "encoding_format": "float",
            },
        )

        assert response.status_code == 200
        assert response.headers["X-Backend-ID"] == "backend-a"
        payload = response.json()
        assert payload["model"] == "lab-embedding"
        assert [item["index"] for item in payload["data"]] == [0, 1]
        assert all(len(item["embedding"]) == 3 for item in payload["data"])
        assert payload["usage"] == {"prompt_tokens": 2, "total_tokens": 2}

        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "backend_id": "backend-a",
            "model": "lab-embedding",
            "dimensions": 3,
            "active_requests": 0,
            "max_concurrency": 2,
        }


async def test_overload_is_rejected_immediately_and_capacity_is_released() -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.15,
        max_concurrency=1,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)
    body = {
        "model": "lab-embedding",
        "input": ["alpha"],
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first_request = asyncio.create_task(client.post("/v1/embeddings", json=body))
        try:
            async with asyncio.timeout(2.0):
                while True:
                    health = await client.get("/health")
                    assert health.status_code == 200
                    if health.json()["active_requests"] == 1:
                        break
                    await asyncio.sleep(0.005)

            rejected = await client.post("/v1/embeddings", json=body)
            assert rejected.status_code == 429
            assert rejected.headers["X-Backend-ID"] == "backend-a"
            assert rejected.headers["Retry-After"] == "1"
            assert rejected.json()["error"] == {
                "message": "model simulator is at capacity",
                "type": "capacity_exhausted",
                "code": "capacity_exhausted",
            }
            assert not first_request.done()

            first_response = await first_request
            assert first_response.status_code == 200
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json()["active_requests"] == 0

            later = await client.post("/v1/embeddings", json=body)
            assert later.status_code == 200
            assert later.headers["X-Backend-ID"] == "backend-a"
        finally:
            if not first_request.done():
                await first_request


async def test_startup_configured_failures_are_deterministic() -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.0,
        max_concurrency=2,
        fail_every_n_requests=2,
        injected_error_status=503,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)
    valid_body = {
        "model": "lab-embedding",
        "input": ["alpha"],
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/health")
        assert health.status_code == 200

        invalid = await client.post(
            "/v1/embeddings",
            json={
                "model": "lab-embedding",
                "input": [],
                "encoding_format": "float",
            },
        )
        assert invalid.status_code == 400

        responses = [await client.post("/v1/embeddings", json=valid_body) for _ in range(4)]
        assert [response.status_code for response in responses] == [200, 503, 200, 503]

        expected_error = {
            "message": "deterministic injected failure",
            "type": "injected_error",
            "code": "injected_error",
        }
        for response in (responses[1], responses[3]):
            assert response.headers["X-Backend-ID"] == "backend-a"
            assert response.json()["error"] == expected_error

        health = await client.get("/health")
        assert health.json()["active_requests"] == 0


@pytest.mark.parametrize(
    ("payload", "raw_content", "expected_message"),
    [
        (
            {
                "model": "wrong-model",
                "input": ["alpha"],
                "encoding_format": "float",
            },
            None,
            "model must match the configured model",
        ),
        (
            {"model": "lab-embedding", "input": [], "encoding_format": "float"},
            None,
            "input must be a nonempty list of strings",
        ),
        (
            {"model": "lab-embedding", "input": "alpha", "encoding_format": "float"},
            None,
            "input must be a nonempty list of strings",
        ),
        (
            {"model": "lab-embedding", "input": ["alpha", 3], "encoding_format": "float"},
            None,
            "input must be a nonempty list of strings",
        ),
        (
            {"model": "lab-embedding", "input": ["alpha"], "encoding_format": "base64"},
            None,
            'encoding_format must be "float"',
        ),
        (
            {"model": "lab-embedding", "input": ["alpha"]},
            None,
            'encoding_format must be "float"',
        ),
        (["not", "an", "object"], None, "request body must be a JSON object"),
        (None, b"{", "request body must be valid JSON"),
    ],
)
async def test_invalid_requests_return_bounded_errors_without_consuming_capacity(
    payload: object | None,
    raw_content: bytes | None,
    expected_message: str,
) -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.0,
        max_concurrency=1,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)
    valid_body = {
        "model": "lab-embedding",
        "input": ["alpha"],
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        if raw_content is None:
            invalid = await client.post("/v1/embeddings", json=payload)
        else:
            invalid = await client.post(
                "/v1/embeddings",
                content=raw_content,
                headers={"Content-Type": "application/json"},
            )

        assert invalid.status_code == 400
        assert invalid.headers["X-Backend-ID"] == "backend-a"
        error = invalid.json()["error"]
        assert set(error) == {"message", "type", "code"}
        assert error["message"] == expected_message
        assert len(error["message"]) <= 500
        assert error["type"] == "BadRequestError"
        assert error["code"] == 400

        valid = await client.post("/v1/embeddings", json=valid_body)
        assert valid.status_code == 200
        assert valid.headers["X-Backend-ID"] == "backend-a"


@pytest.mark.parametrize(
    ("content_length", "expected_status", "expected_message"),
    [
        (
            str(MAX_REQUEST_BYTES + 1),
            413,
            f"request body must not exceed {MAX_REQUEST_BYTES} bytes",
        ),
        ("not-an-integer", 400, "content-length must be an integer"),
    ],
)
async def test_invalid_content_length_is_rejected_before_work_is_admitted(
    content_length: str,
    expected_status: int,
    expected_message: str,
) -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.0,
        max_concurrency=1,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/embeddings",
            content=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": content_length},
        )

        assert response.status_code == expected_status
        assert response.json()["error"]["message"] == expected_message
        health = await client.get("/health")
        assert health.json()["active_requests"] == 0


async def test_cancelled_request_releases_capacity() -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.1,
        max_concurrency=1,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)
    body = {
        "model": "lab-embedding",
        "input": ["alpha"],
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        request = asyncio.create_task(client.post("/v1/embeddings", json=body))

        async with asyncio.timeout(2.0):
            while True:
                health = await client.get("/health")
                assert health.status_code == 200
                if health.json()["active_requests"] == 1:
                    break
                await asyncio.sleep(0.005)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        async with asyncio.timeout(2.0):
            while True:
                health = await client.get("/health")
                assert health.status_code == 200
                if health.json()["active_requests"] == 0:
                    break
                await asyncio.sleep(0.005)

        later = await client.post("/v1/embeddings", json=body)
        assert later.status_code == 200
        assert later.headers["X-Backend-ID"] == "backend-a"


@pytest.mark.parametrize("case", ["request_bytes", "input_items", "text_length"])
async def test_safety_bounds_reject_oversized_requests_without_consuming_capacity(
    case: str,
) -> None:
    settings = ModelSimulatorSettings(
        backend_id="backend-a",
        model="lab-embedding",
        dimensions=3,
        delay_seconds=0.0,
        max_concurrency=1,
    )
    app = create_app(settings, SyntheticEmbeddingService(dimensions=settings.dimensions))
    transport = httpx.ASGITransport(app=app)
    valid_body = {
        "model": "lab-embedding",
        "input": ["alpha"],
        "encoding_format": "float",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        if case == "request_bytes":
            raw_body = (
                b'{"model":"lab-embedding","input":["'
                + b"a" * MAX_REQUEST_BYTES
                + b'"],"encoding_format":"float"}'
            )
            rejected = await client.post(
                "/v1/embeddings",
                content=raw_body,
                headers={"Content-Type": "application/json"},
            )
            assert rejected.status_code == 413
            assert rejected.json()["error"]["type"] == "request_too_large"
            assert rejected.json()["error"]["code"] == "request_too_large"
        elif case == "input_items":
            rejected = await client.post(
                "/v1/embeddings",
                json={
                    "model": "lab-embedding",
                    "input": ["alpha"] * (MAX_INPUT_ITEMS + 1),
                    "encoding_format": "float",
                },
            )
            assert rejected.status_code == 400
        else:
            rejected = await client.post(
                "/v1/embeddings",
                json={
                    "model": "lab-embedding",
                    "input": ["a" * (MAX_TEXT_LENGTH + 1)],
                    "encoding_format": "float",
                },
            )
            assert rejected.status_code == 400

        assert rejected.headers["X-Backend-ID"] == "backend-a"
        assert len(rejected.json()["error"]["message"]) <= 500

        valid = await client.post("/v1/embeddings", json=valid_body)
        assert valid.status_code == 200
        assert valid.headers["X-Backend-ID"] == "backend-a"
