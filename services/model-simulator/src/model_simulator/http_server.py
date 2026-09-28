"""OpenAI-compatible embedding surface for the synthetic model simulator."""

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_simulator.ports import EmbeddingService

MAX_REQUEST_BYTES = 1_048_576
MAX_INPUT_ITEMS = 256
MAX_TEXT_LENGTH = 8_192


@dataclass(frozen=True, slots=True)
class ModelSimulatorSettings:
    """Settings for one local fake embedding backend."""

    backend_id: str
    model: str
    dimensions: int
    delay_seconds: float
    max_concurrency: int
    fail_every_n_requests: int | None = None
    injected_error_status: int = 503

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError("backend_id must not be blank")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be blank")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValueError("dimensions must be a positive integer")
        if (
            isinstance(self.delay_seconds, bool)
            or not isinstance(self.delay_seconds, (int, float))
            or not math.isfinite(self.delay_seconds)
            or self.delay_seconds < 0
        ):
            raise ValueError("delay_seconds must be finite and nonnegative")
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if self.fail_every_n_requests is not None and (
            isinstance(self.fail_every_n_requests, bool)
            or not isinstance(self.fail_every_n_requests, int)
            or self.fail_every_n_requests < 1
        ):
            raise ValueError("fail_every_n_requests must be a positive integer or None")
        if (
            isinstance(self.injected_error_status, bool)
            or not isinstance(self.injected_error_status, int)
            or not 500 <= self.injected_error_status <= 599
        ):
            raise ValueError("injected_error_status must be an HTTP 5xx status")


def create_app(settings: ModelSimulatorSettings, service: EmbeddingService) -> FastAPI:
    """Create a small OpenAI-subset app backed by deterministic synthetic vectors."""

    app = FastAPI()
    active_requests = 0
    accepted_requests = 0
    active_requests_lock = asyncio.Lock()

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request) -> JSONResponse:
        nonlocal accepted_requests, active_requests

        payload, body_error = await _read_json(request, settings.backend_id)
        if body_error is not None:
            return body_error

        if not isinstance(payload, dict):
            return _bad_request("request body must be a JSON object", settings.backend_id)
        if payload.get("model") != settings.model:
            return _bad_request("model must match the configured model", settings.backend_id)

        texts = payload.get("input")
        if not isinstance(texts, list) or not texts:
            return _bad_request("input must be a nonempty list of strings", settings.backend_id)
        if any(not isinstance(text, str) for text in texts):
            return _bad_request("input must be a nonempty list of strings", settings.backend_id)
        if len(texts) > MAX_INPUT_ITEMS:
            return _bad_request(
                f"input must contain at most {MAX_INPUT_ITEMS} items", settings.backend_id
            )
        if any(len(text) > MAX_TEXT_LENGTH for text in texts):
            return _bad_request(
                f"each input string must contain at most {MAX_TEXT_LENGTH} characters",
                settings.backend_id,
            )
        if payload.get("encoding_format") != "float":
            return _bad_request('encoding_format must be "float"', settings.backend_id)

        async with active_requests_lock:
            if active_requests >= settings.max_concurrency:
                return _capacity_exhausted(settings.backend_id)
            active_requests += 1
            accepted_requests += 1
            inject_failure = (
                settings.fail_every_n_requests is not None
                and accepted_requests % settings.fail_every_n_requests == 0
            )
        try:
            await asyncio.sleep(settings.delay_seconds)
            if inject_failure:
                return _injected_error(settings)
            result = await service.embed(texts)
        finally:
            async with active_requests_lock:
                active_requests -= 1

        return JSONResponse(
            content={
                "object": "list",
                "model": settings.model,
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": list(embedding.values),
                    }
                    for index, embedding in enumerate(result.embeddings)
                ],
                "usage": {
                    "prompt_tokens": result.input_tokens,
                    "total_tokens": result.input_tokens,
                },
            },
            headers={"X-Backend-ID": settings.backend_id},
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        async with active_requests_lock:
            current_active_requests = active_requests
        return {
            "status": "ok",
            "backend_id": settings.backend_id,
            "model": settings.model,
            "dimensions": settings.dimensions,
            "active_requests": current_active_requests,
            "max_concurrency": settings.max_concurrency,
        }

    return app


async def _read_json(request: Request, backend_id: str) -> tuple[Any, JSONResponse | None]:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return None, _request_too_large(backend_id)
        except ValueError:
            return None, _bad_request("content-length must be an integer", backend_id)

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            return None, _request_too_large(backend_id)
        body.extend(chunk)

    try:
        return json.loads(body), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, _bad_request("request body must be valid JSON", backend_id)


def _bad_request(message: str, backend_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "BadRequestError",
                "code": 400,
            }
        },
        headers={"X-Backend-ID": backend_id},
    )


def _request_too_large(backend_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "error": {
                "message": f"request body must not exceed {MAX_REQUEST_BYTES} bytes",
                "type": "request_too_large",
                "code": "request_too_large",
            }
        },
        headers={"X-Backend-ID": backend_id},
    )


def _capacity_exhausted(backend_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "model simulator is at capacity",
                "type": "capacity_exhausted",
                "code": "capacity_exhausted",
            }
        },
        headers={"X-Backend-ID": backend_id, "Retry-After": "1"},
    )


def _injected_error(settings: ModelSimulatorSettings) -> JSONResponse:
    return JSONResponse(
        status_code=settings.injected_error_status,
        content={
            "error": {
                "message": "deterministic injected failure",
                "type": "injected_error",
                "code": "injected_error",
            }
        },
        headers={"X-Backend-ID": settings.backend_id},
    )
