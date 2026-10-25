"""Atomic backend reservation and replaceable selection strategies."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from gateway.domain import BackendId, EmbedCommand, EmbeddingResponse, RequestContext
from gateway.failures import NoUsableBackend


class BackendClient(Protocol):
    """Outbound seam implemented by one long-lived client per backend."""

    @property
    def backend_id(self) -> BackendId: ...

    async def embed(self, command: EmbedCommand, context: RequestContext) -> EmbeddingResponse: ...

    async def is_ready(self) -> bool: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BackendSnapshot:
    """Only the scheduling facts a selection strategy may inspect."""

    backend_id: BackendId
    outstanding: int


class BackendSelectionStrategy(Protocol):
    """Choose exactly one backend from a nonempty eligible sequence."""

    def select(self, candidates: Sequence[BackendSnapshot]) -> BackendId: ...


class RoundRobinSelection:
    """Rotate through eligible backends in their configured order."""

    def __init__(self) -> None:
        self._next_index = 0

    def select(self, candidates: Sequence[BackendSnapshot]) -> BackendId:
        if not candidates:
            raise NoUsableBackend
        selected = candidates[self._next_index % len(candidates)]
        self._next_index = (self._next_index + 1) % len(candidates)
        return selected.backend_id


@dataclass(slots=True)
class _BackendState:
    client: BackendClient
    outstanding: int = 0


@dataclass(frozen=True, slots=True)
class BackendReservation:
    """A private lease returned only inside the pool context manager."""

    backend_id: BackendId
    client: BackendClient


class BackendPool:
    """Keep selection, reservation, and release atomic and local."""

    def __init__(
        self,
        clients: Sequence[BackendClient],
        selection: BackendSelectionStrategy,
    ) -> None:
        if not clients:
            raise ValueError("at least one backend is required")
        backend_ids = [client.backend_id for client in clients]
        if len(set(backend_ids)) != len(backend_ids):
            raise ValueError("backend IDs must be unique")
        self._states = [_BackendState(client=client) for client in clients]
        self._selection = selection
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[BackendReservation]:
        async with self._lock:
            snapshots = tuple(
                BackendSnapshot(
                    backend_id=state.client.backend_id,
                    outstanding=state.outstanding,
                )
                for state in self._states
            )
            selected_id = self._selection.select(snapshots)
            selected = next(
                (state for state in self._states if state.client.backend_id == selected_id),
                None,
            )
            if selected is None:
                raise RuntimeError("selection strategy returned an unknown backend")
            selected.outstanding += 1

        try:
            yield BackendReservation(
                backend_id=selected.client.backend_id,
                client=selected.client,
            )
        finally:
            async with self._lock:
                selected.outstanding -= 1

    async def aclose(self) -> None:
        await asyncio.gather(*(state.client.aclose() for state in self._states))

    async def is_ready(self) -> bool:
        readiness = await asyncio.gather(
            *(state.client.is_ready() for state in self._states),
            return_exceptions=True,
        )
        return any(result is True for result in readiness)
