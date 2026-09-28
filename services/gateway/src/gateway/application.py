"""Deep application module for the embedding routing use case."""

import asyncio
import time

from gateway.domain import BackendId, EmbedCommand, RequestContext, RoutedEmbeddingResult
from gateway.failures import (
    DeadlineExceeded,
    GatewayFailure,
    InvalidUpstreamResponse,
    UnknownModel,
)
from gateway.routing import BackendPool
from gateway.telemetry import GatewayTelemetry, NoopGatewayTelemetry


class Gateway:
    """Validate, route, forward once, validate the result, and clean up."""

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        pool: BackendPool,
        telemetry: GatewayTelemetry | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._model = model
        self._dimensions = dimensions
        self._pool = pool
        self._telemetry = telemetry or NoopGatewayTelemetry()

    async def embed(self, command: EmbedCommand, context: RequestContext) -> RoutedEmbeddingResult:
        if command.model != self._model:
            raise UnknownModel

        backend_id: BackendId | None = None
        upstream_started: float | None = None
        upstream_outcome = "internal_error"
        try:
            async with asyncio.timeout_at(context.deadline):
                async with self._pool.reserve() as reservation:
                    backend_id = reservation.backend_id
                    self._telemetry.backend_selected(backend_id=reservation.backend_id)
                    self._telemetry.upstream_started(backend_id=reservation.backend_id)
                    upstream_started = time.monotonic()
                    response = await reservation.client.embed(command, context)
                    self._validate_response(response, expected_count=len(command.inputs))
                    upstream_outcome = "succeeded"
                    return RoutedEmbeddingResult(
                        backend_id=reservation.backend_id,
                        response=response,
                    )
        except TimeoutError as error:
            upstream_outcome = "deadline_exceeded"
            raise DeadlineExceeded(backend_id=backend_id) from error
        except asyncio.CancelledError:
            upstream_outcome = "cancelled"
            raise
        except GatewayFailure as failure:
            upstream_outcome = failure.code
            if failure.backend_id is None:
                failure.backend_id = backend_id
            raise
        finally:
            if backend_id is not None and upstream_started is not None:
                self._telemetry.upstream_completed(
                    backend_id=backend_id,
                    outcome=upstream_outcome,
                    duration_seconds=time.monotonic() - upstream_started,
                )

    def _validate_response(self, response: object, *, expected_count: int) -> None:
        from gateway.domain import EmbeddingResponse

        if not isinstance(response, EmbeddingResponse):
            raise InvalidUpstreamResponse
        if response.model != self._model:
            raise InvalidUpstreamResponse
        if len(response.embeddings) != expected_count:
            raise InvalidUpstreamResponse
        if tuple(item.index for item in response.embeddings) != tuple(range(expected_count)):
            raise InvalidUpstreamResponse
        if any(len(item.values) != self._dimensions for item in response.embeddings):
            raise InvalidUpstreamResponse

    async def aclose(self) -> None:
        await self._pool.aclose()

    async def is_ready(self) -> bool:
        return await self._pool.is_ready()
