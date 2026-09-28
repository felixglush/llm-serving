"""FastAPI adapter for the gateway's public HTTP interface."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gateway.application import Gateway
from gateway.domain import EmbedCommand, RequestContext, RoutedEmbeddingResult
from gateway.failures import (
    DeadlineExceeded,
    GatewayFailure,
    InvalidRequest,
    InvalidUpstreamResponse,
    NoUsableBackend,
    RequestTooLarge,
    UnknownModel,
    UpstreamUnavailable,
)
from gateway.telemetry import GatewayTelemetry, NoopGatewayTelemetry

LOGGER = logging.getLogger("gateway.requests")
MAX_REQUEST_BYTES = 1_048_576
MAX_INPUT_ITEMS = 256
MAX_TEXT_LENGTH = 8_192
EMBEDDINGS_ROUTE = "/v1/embeddings"


def create_app(
    *,
    gateway: Gateway,
    request_timeout_seconds: float,
    telemetry: GatewayTelemetry | None = None,
) -> FastAPI:
    """Create the public HTTP adapter around one gateway use case."""

    request_telemetry = telemetry or NoopGatewayTelemetry()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await gateway.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.post(EMBEDDINGS_ROUTE)
    async def create_embeddings(request: Request) -> JSONResponse:
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        deadline = asyncio.get_running_loop().time() + request_timeout_seconds
        backend_id: str | None = None
        outcome = "internal_error"
        request_telemetry.request_received(route=EMBEDDINGS_ROUTE)
        try:
            async with asyncio.timeout_at(deadline):
                payload = await _read_json(request)
                command = _parse_command(payload)
            context = RequestContext(request_id=request_id, deadline=deadline)
            result = await gateway.embed(command, context)
            backend_id = result.backend_id
            outcome = "succeeded"
            return _success_response(result, request_id=request_id)
        except TimeoutError:
            failure = DeadlineExceeded(backend_id=backend_id)
            outcome = failure.code
            return _error_response(failure, request_id=request_id)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except GatewayFailure as failure:
            backend_id = failure.backend_id
            outcome = failure.code
            if isinstance(failure, (InvalidRequest, RequestTooLarge, UnknownModel)):
                request_telemetry.request_rejected(
                    route=EMBEDDINGS_ROUTE,
                    reason=failure.code,
                )
            return _error_response(failure, request_id=request_id)
        finally:
            duration_seconds = time.monotonic() - started
            request_telemetry.request_completed(
                route=EMBEDDINGS_ROUTE,
                outcome=outcome,
                duration_seconds=duration_seconds,
            )
            LOGGER.info(
                json.dumps(
                    {
                        "event": "embedding_request_completed",
                        "request_id": request_id,
                        "backend_id": backend_id,
                        "outcome": outcome,
                        "duration_seconds": duration_seconds,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        if await gateway.is_ready():
            return JSONResponse(content={"status": "ready"})
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    return app


def _parse_command(payload: Any) -> EmbedCommand:
    if not isinstance(payload, dict):
        raise InvalidRequest
    model = payload.get("model")
    inputs = payload.get("input")
    encoding_format = payload.get("encoding_format")
    if not isinstance(model, str) or not model.strip():
        raise InvalidRequest
    if not isinstance(inputs, list) or not inputs or any(not isinstance(v, str) for v in inputs):
        raise InvalidRequest
    if len(inputs) > MAX_INPUT_ITEMS:
        raise InvalidRequest(f"input must contain at most {MAX_INPUT_ITEMS} items")
    if any(len(value) > MAX_TEXT_LENGTH for value in inputs):
        raise InvalidRequest(f"each input string must contain at most {MAX_TEXT_LENGTH} characters")
    if encoding_format != "float":
        raise InvalidRequest
    return EmbedCommand(model=model, inputs=tuple(inputs), encoding_format="float")


async def _read_json(request: Request) -> Any:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise RequestTooLarge(f"request body must not exceed {MAX_REQUEST_BYTES} bytes")
        except ValueError as error:
            raise InvalidRequest("content-length must be an integer") from error

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise RequestTooLarge(f"request body must not exceed {MAX_REQUEST_BYTES} bytes")
        body.extend(chunk)

    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidRequest("request body must be valid JSON") from error


def _success_response(result: RoutedEmbeddingResult, *, request_id: str) -> JSONResponse:
    response = result.response
    return JSONResponse(
        content={
            "object": "list",
            "model": response.model,
            "data": [
                {
                    "object": "embedding",
                    "index": item.index,
                    "embedding": list(item.values),
                }
                for item in response.embeddings
            ],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "total_tokens": response.total_tokens,
            },
        },
        headers={
            "X-Request-ID": request_id,
            "X-Backend-ID": result.backend_id,
        },
    )


def _error_response(failure: GatewayFailure, *, request_id: str) -> JSONResponse:
    if isinstance(failure, (InvalidRequest, UnknownModel)):
        status_code = 400
    elif isinstance(failure, RequestTooLarge):
        status_code = 413
    elif isinstance(failure, InvalidUpstreamResponse):
        status_code = 502
    elif isinstance(failure, DeadlineExceeded):
        status_code = 504
    elif isinstance(failure, (NoUsableBackend, UpstreamUnavailable)):
        status_code = 503
    else:
        status_code = 500
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": failure.safe_message,
                "type": failure.code,
                "code": failure.code,
            }
        },
        headers={"X-Request-ID": request_id},
    )
