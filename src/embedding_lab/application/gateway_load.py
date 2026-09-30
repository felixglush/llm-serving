"""Bounded open-loop gateway load scheduling and local persistence."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from embedding_lab.domain.gateway_load import (
    FixedArrivalPlan,
    GatewayLoadObservation,
    GatewayLoadResult,
    LoadOutcome,
)
from embedding_lab.ports.gateway_client import (
    GatewayClient,
    GatewayClientError,
    GatewayHttpError,
    GatewayProtocolError,
    GatewayRequestTimeout,
    GatewayTransportError,
)

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class _DispatchedRequest:
    request_index: int
    intended_offset: float
    actual_offset: float | None = None
    dispatch_lag_ms: float | None = None


class RunGatewayLoad:
    """Dispatch fixed arrivals without waiting for earlier request completions."""

    def __init__(
        self,
        client: GatewayClient,
        *,
        clock: Clock = time.perf_counter,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._client = client
        self._clock = clock
        self._sleep = sleep

    async def run(self, plan: FixedArrivalPlan) -> GatewayLoadResult:
        started = self._clock()
        observations: dict[int, GatewayLoadObservation] = {}
        pending: dict[asyncio.Task[GatewayLoadObservation], _DispatchedRequest] = {}
        max_observed_outstanding = 0
        offsets = plan.arrival_offsets
        try:
            for position, intended_offset in enumerate(offsets):
                await self._sleep(max(started + intended_offset - self._clock(), 0.0))
                self._collect_completed(pending, observations)
                if len(pending) >= plan.max_outstanding:
                    observations[position] = GatewayLoadObservation(
                        request_index=position,
                        intended_dispatch_offset_seconds=intended_offset,
                        actual_dispatch_offset_seconds=None,
                        completion_offset_seconds=None,
                        dispatch_lag_ms=None,
                        latency_ms=None,
                        outcome="generator_dropped",
                        service_error_code="max_outstanding",
                        error="load generator max_outstanding limit reached",
                    )
                    continue

                metadata = _DispatchedRequest(
                    request_index=position,
                    intended_offset=intended_offset,
                )
                texts = list(plan.texts_for_request(position))
                task = asyncio.create_task(
                    self._request(
                        texts,
                        metadata=metadata,
                        run_started=started,
                    )
                )
                pending[task] = metadata
                max_observed_outstanding = max(max_observed_outstanding, len(pending))
                if position < len(offsets) - 1:
                    await self._sleep(0)

            arrival_phase_elapsed = max(self._clock() - started, 0.0)
            if pending:
                done, unfinished = await asyncio.wait(
                    pending,
                    timeout=plan.drain_timeout_seconds,
                )
                for task in done:
                    observations[task.result().request_index] = task.result()
                    pending.pop(task)
                if unfinished:
                    completion_offset = max(self._clock() - started, 0.0)
                    for task in unfinished:
                        metadata = pending.pop(task)
                        task.cancel()
                        actual_offset = metadata.actual_offset
                        if actual_offset is None:
                            actual_offset = max(self._clock() - started, 0.0)
                        dispatch_lag_ms = metadata.dispatch_lag_ms
                        if dispatch_lag_ms is None:
                            dispatch_lag_ms = max(
                                (actual_offset - metadata.intended_offset) * 1_000,
                                0.0,
                            )
                        observations[metadata.request_index] = GatewayLoadObservation(
                            request_index=metadata.request_index,
                            intended_dispatch_offset_seconds=metadata.intended_offset,
                            actual_dispatch_offset_seconds=actual_offset,
                            completion_offset_seconds=completion_offset,
                            dispatch_lag_ms=dispatch_lag_ms,
                            latency_ms=max(
                                (completion_offset - actual_offset) * 1_000,
                                0.0,
                            ),
                            outcome="drain_timeout",
                            error="request did not finish before drain timeout",
                        )
                    await asyncio.gather(*unfinished, return_exceptions=True)
            self._collect_completed(pending, observations)
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise

        total_elapsed = max(self._clock() - started, 1e-12)
        return GatewayLoadResult(
            target_request_rate=plan.request_rate,
            arrival_window_seconds=min(
                plan.duration_seconds,
                len(offsets) / plan.request_rate,
            ),
            arrival_phase_elapsed_seconds=arrival_phase_elapsed,
            total_elapsed_seconds=total_elapsed,
            max_observed_outstanding=max_observed_outstanding,
            observations=tuple(observations[index] for index in range(len(offsets))),
        )

    def _collect_completed(
        self,
        pending: dict[asyncio.Task[GatewayLoadObservation], _DispatchedRequest],
        observations: dict[int, GatewayLoadObservation],
    ) -> None:
        for task in tuple(pending):
            if task.done():
                observation = task.result()
                observations[observation.request_index] = observation
                pending.pop(task)

    async def _request(
        self,
        texts: list[str],
        *,
        metadata: _DispatchedRequest,
        run_started: float,
    ) -> GatewayLoadObservation:
        actual_offset = max(self._clock() - run_started, 0.0)
        metadata.actual_offset = actual_offset
        metadata.dispatch_lag_ms = max(
            (actual_offset - metadata.intended_offset) * 1_000,
            0.0,
        )
        outcome: LoadOutcome = "client_error"
        status_code: int | None = None
        backend_id: str | None = None
        service_error_code: str | None = None
        error_message: str | None = None
        vectors = 0
        input_tokens = 0
        try:
            response = await self._client.embed(texts)
            outcome = "succeeded"
            status_code = 200
            backend_id = response.backend_id
            vectors = response.vectors
            input_tokens = response.input_tokens
        except GatewayHttpError as error:
            outcome = "http_error"
            status_code = error.status_code
            service_error_code = error.error_code
            error_message = str(error)
        except GatewayRequestTimeout as error:
            outcome = "timeout"
            error_message = str(error)
        except GatewayTransportError as error:
            outcome = "transport_error"
            error_message = str(error)
        except GatewayProtocolError as error:
            outcome = "protocol_error"
            error_message = str(error)
        except GatewayClientError as error:
            outcome = "client_error"
            error_message = str(error)
        completion_offset = max(self._clock() - run_started, actual_offset)
        return GatewayLoadObservation(
            request_index=metadata.request_index,
            intended_dispatch_offset_seconds=metadata.intended_offset,
            actual_dispatch_offset_seconds=actual_offset,
            completion_offset_seconds=completion_offset,
            dispatch_lag_ms=metadata.dispatch_lag_ms,
            latency_ms=max((completion_offset - actual_offset) * 1_000, 0.0),
            outcome=outcome,
            status_code=status_code,
            backend_id=backend_id,
            service_error_code=service_error_code,
            error=error_message,
            vectors=vectors,
            input_tokens=input_tokens,
        )
