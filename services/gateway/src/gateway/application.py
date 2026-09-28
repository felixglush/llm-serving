"""Deep application module for the embedding routing use case."""

import asyncio

from gateway.domain import EmbedCommand, RequestContext, RoutedEmbeddingResult
from gateway.failures import (
    DeadlineExceeded,
    GatewayFailure,
    InvalidUpstreamResponse,
    UnknownModel,
)
from gateway.routing import BackendPool


class Gateway:
    """Validate, route, forward once, validate the result, and clean up."""

    def __init__(self, *, model: str, dimensions: int, pool: BackendPool) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._model = model
        self._dimensions = dimensions
        self._pool = pool

    async def embed(self, command: EmbedCommand, context: RequestContext) -> RoutedEmbeddingResult:
        if command.model != self._model:
            raise UnknownModel

        backend_id: str | None = None
        try:
            async with asyncio.timeout_at(context.deadline):
                async with self._pool.reserve() as reservation:
                    backend_id = reservation.backend_id
                    response = await reservation.client.embed(command, context)
                    self._validate_response(response, expected_count=len(command.inputs))
                    return RoutedEmbeddingResult(
                        backend_id=reservation.backend_id,
                        response=response,
                    )
        except TimeoutError as error:
            raise DeadlineExceeded(backend_id=backend_id) from error
        except GatewayFailure as failure:
            if failure.backend_id is None:
                failure.backend_id = backend_id
            raise

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
