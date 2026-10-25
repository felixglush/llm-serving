import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from embedding_lab.adapters.filesystem.gateway_load_store import GatewayLoadStore
from embedding_lab.application.gateway_load import RunGatewayLoad
from embedding_lab.domain.gateway_load import (
    FixedArrivalPlan,
    GatewayLoadClientConfig,
    GatewayLoadExperiment,
    GatewayLoadTarget,
)
from embedding_lab.ports.gateway_client import (
    GatewayCallResult,
    GatewayClientError,
    GatewayHttpError,
    GatewayProtocolError,
    GatewayRequestTimeout,
    GatewayTransportError,
)


class ManualClock:
    def __init__(self, *, overshoot_seconds: float = 0.0) -> None:
        self.now = 100.0
        self.overshoot_seconds = overshoot_seconds

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay_seconds: float) -> None:
        self.now += max(delay_seconds, 0.0)
        if delay_seconds > 0:
            self.now += self.overshoot_seconds
        await asyncio.sleep(0)


class ScriptedGatewayClient:
    def __init__(
        self,
        action: Callable[[int, list[str]], Awaitable[GatewayCallResult]],
    ) -> None:
        self._action = action
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def embed(self, texts: list[str]) -> GatewayCallResult:
        call = self.calls
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return await self._action(call, texts)
        finally:
            self.active -= 1


def plan(**overrides: object) -> FixedArrivalPlan:
    values: dict[str, object] = {
        "request_rate": 10.0,
        "duration_seconds": 1.0,
        "max_requests": 3,
        "max_outstanding": 3,
        "drain_timeout_seconds": 0.05,
        "texts_per_request": 1,
        "target_tokens_per_text": 3,
        "seed": 7,
        "corpus_version": "gateway-load-fixture-v1",
    }
    values.update(overrides)
    return FixedArrivalPlan(**values)  # type: ignore[arg-type]


async def test_fixed_arrivals_are_dispatched_independently_of_completions() -> None:
    release = asyncio.Event()

    async def complete_after_every_request_started(
        call: int, texts: list[str]
    ) -> GatewayCallResult:
        assert len(texts) == 1
        if call == 2:
            release.set()
        await release.wait()
        return GatewayCallResult(
            backend_id="backend-a" if call % 2 == 0 else "backend-b",
            vectors=len(texts),
            input_tokens=len(texts),
        )

    clock = ManualClock()
    client = ScriptedGatewayClient(complete_after_every_request_started)

    result = await RunGatewayLoad(client, clock=clock, sleep=clock.sleep).run(plan())

    assert client.calls == 3
    assert client.max_active == 3
    assert result.scheduled_requests == 3
    assert result.dispatched_requests == 3
    assert result.successful_requests == 3
    assert result.generator_dropped_requests == 0
    assert result.max_observed_outstanding == 3
    assert result.backend_counts == {"backend-a": 2, "backend-b": 1}
    assert [item.intended_dispatch_offset_seconds for item in result.observations] == [
        0.0,
        0.1,
        0.2,
    ]
    assert [item.actual_dispatch_offset_seconds for item in result.observations] == pytest.approx(
        [0.0, 0.1, 0.2]
    )
    assert all(item.dispatch_lag_ms == pytest.approx(0.0, abs=1e-6) for item in result.observations)


async def test_outstanding_cap_records_generator_drops_without_waiting_for_capacity() -> None:
    never_release = asyncio.Event()

    async def block(_: int, __: list[str]) -> GatewayCallResult:
        await never_release.wait()
        raise AssertionError("unreachable")

    clock = ManualClock()
    client = ScriptedGatewayClient(block)

    result = await RunGatewayLoad(client, clock=clock, sleep=clock.sleep).run(
        plan(max_requests=5, max_outstanding=2, drain_timeout_seconds=0.01)
    )

    assert client.calls == 2
    assert client.active == 0
    assert result.scheduled_requests == 5
    assert result.dispatched_requests == 2
    assert result.generator_dropped_requests == 3
    assert result.drain_timed_out_requests == 2
    assert result.outcome_counts == {"drain_timeout": 2, "generator_dropped": 3}
    assert [item.request_index for item in result.observations] == [0, 1, 2, 3, 4]
    assert result.max_observed_outstanding == 2


async def test_released_capacity_only_allows_future_arrivals_and_never_replays_a_drop() -> None:
    release = asyncio.Event()

    class ReleasingClock(ManualClock):
        async def sleep(self, delay_seconds: float) -> None:
            await super().sleep(delay_seconds)
            if self.now >= 100.15:
                release.set()
                await asyncio.sleep(0)

    async def wait_until_released(call: int, texts: list[str]) -> GatewayCallResult:
        if call == 0:
            await release.wait()
        return GatewayCallResult(
            backend_id="backend-a",
            vectors=len(texts),
            input_tokens=len(texts),
        )

    clock = ReleasingClock()
    client = ScriptedGatewayClient(wait_until_released)
    result = await RunGatewayLoad(client, clock=clock, sleep=clock.sleep).run(
        plan(max_requests=3, max_outstanding=1)
    )

    assert client.calls == 2
    assert [item.outcome for item in result.observations] == [
        "succeeded",
        "generator_dropped",
        "succeeded",
    ]
    assert result.observations[1].actual_dispatch_offset_seconds is None
    assert result.observations[2].actual_dispatch_offset_seconds == pytest.approx(0.2)


async def test_scheduler_lag_is_measured_instead_of_hidden() -> None:
    async def succeed(_: int, texts: list[str]) -> GatewayCallResult:
        return GatewayCallResult(backend_id="backend-a", vectors=len(texts), input_tokens=1)

    clock = ManualClock(overshoot_seconds=0.025)
    result = await RunGatewayLoad(
        ScriptedGatewayClient(succeed), clock=clock, sleep=clock.sleep
    ).run(plan())

    assert [item.dispatch_lag_ms for item in result.observations] == pytest.approx(
        [0.0, 25.0, 25.0]
    )
    assert result.max_dispatch_lag_ms == pytest.approx(25.0)
    assert result.generator_dropped_requests == 0


async def test_dispatch_rate_excludes_requests_dispatched_after_arrival_window() -> None:
    async def succeed(_: int, texts: list[str]) -> GatewayCallResult:
        return GatewayCallResult(backend_id="backend-a", vectors=len(texts), input_tokens=1)

    clock = ManualClock(overshoot_seconds=0.75)
    result = await RunGatewayLoad(
        ScriptedGatewayClient(succeed), clock=clock, sleep=clock.sleep
    ).run(plan(request_rate=2.0, duration_seconds=1.0, max_requests=2))

    assert result.dispatched_requests == 2
    assert result.in_window_dispatched_requests == 1
    assert result.dispatch_shortfall_requests == 1
    assert result.actual_dispatch_rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("error", "outcome", "status_code", "service_error_code"),
    [
        (
            GatewayHttpError(429, "capacity_exceeded", "full"),
            "http_error",
            429,
            "capacity_exceeded",
        ),
        (GatewayRequestTimeout("deadline"), "timeout", None, None),
        (GatewayTransportError("connection refused"), "transport_error", None, None),
        (GatewayProtocolError("invalid response"), "protocol_error", None, None),
        (GatewayClientError("unexpected adapter failure"), "client_error", None, None),
    ],
)
async def test_each_request_failure_class_has_a_distinct_observable_outcome(
    error: Exception,
    outcome: str,
    status_code: int | None,
    service_error_code: str | None,
) -> None:
    async def fail(_: int, __: list[str]) -> GatewayCallResult:
        raise error

    clock = ManualClock()
    result = await RunGatewayLoad(ScriptedGatewayClient(fail), clock=clock, sleep=clock.sleep).run(
        plan(max_requests=1)
    )

    assert result.scheduled_requests == 1
    assert result.dispatched_requests == 1
    assert result.successful_requests == 0
    assert result.failed_requests == 1
    assert result.outcome_counts == {outcome: 1}
    assert result.observations[0].status_code == status_code
    assert result.observations[0].service_error_code == service_error_code
    assert result.observations[0].error
    assert result.p50_latency_ms is None
    assert result.p95_latency_ms is None
    assert result.p99_latency_ms is None


async def test_unexpected_client_defect_aborts_run_and_cancels_sibling_requests() -> None:
    blocked = asyncio.Event()

    async def defect_after_sibling_starts(call: int, _: list[str]) -> GatewayCallResult:
        if call == 0:
            blocked.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        await blocked.wait()
        raise RuntimeError("adapter implementation defect")

    clock = ManualClock()
    client = ScriptedGatewayClient(defect_after_sibling_starts)

    with pytest.raises(RuntimeError, match="adapter implementation defect"):
        await RunGatewayLoad(client, clock=clock, sleep=clock.sleep).run(
            plan(max_requests=2, max_outstanding=2)
        )

    assert client.active == 0


async def test_summary_uses_success_latency_only_and_keeps_load_and_drain_time_separate() -> None:
    clock = ManualClock()

    async def mixed_outcomes(call: int, texts: list[str]) -> GatewayCallResult:
        clock.now += (call + 1) / 100
        if call == 1:
            raise GatewayHttpError(503, "upstream_unavailable", "unavailable")
        return GatewayCallResult(
            backend_id=f"backend-{call}", vectors=len(texts), input_tokens=len(texts)
        )

    result = await RunGatewayLoad(
        ScriptedGatewayClient(mixed_outcomes), clock=clock, sleep=clock.sleep
    ).run(plan(request_rate=1.0, duration_seconds=10.0, max_requests=3))

    assert result.successful_requests == 2
    assert result.failed_requests == 1
    assert result.p50_latency_ms == pytest.approx(20.0)
    assert result.p95_latency_ms == pytest.approx(29.0)
    assert result.p99_latency_ms == pytest.approx(29.8)
    assert result.arrival_phase_elapsed_seconds < result.total_elapsed_seconds
    assert result.whole_run_successful_requests_per_second == pytest.approx(
        2 / result.total_elapsed_seconds
    )
    assert result.actual_dispatch_rate == pytest.approx(1.0)
    assert result.successful_completion_rate_during_arrival_window == pytest.approx(2 / 3)


async def test_cancelling_a_run_cancels_and_drains_every_started_request() -> None:
    started = asyncio.Event()
    never_release = asyncio.Event()

    async def block(_: int, __: list[str]) -> GatewayCallResult:
        started.set()
        await never_release.wait()
        raise AssertionError("unreachable")

    clock = ManualClock()
    client = ScriptedGatewayClient(block)
    task = asyncio.create_task(
        RunGatewayLoad(client, clock=clock, sleep=clock.sleep).run(
            plan(max_requests=2, max_outstanding=2, drain_timeout_seconds=60.0)
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.active == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_rate", 0),
        ("request_rate", math.inf),
        ("duration_seconds", 0),
        ("duration_seconds", math.nan),
        ("max_requests", 0),
        ("max_outstanding", 0),
        ("drain_timeout_seconds", -1),
        ("texts_per_request", 0),
        ("target_tokens_per_text", 0),
        ("seed", True),
        ("corpus_version", " "),
    ],
)
def test_invalid_load_plan_is_rejected_before_a_run(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        plan(**{field: value})


def test_planned_arrivals_are_bounded_by_duration_and_request_limit() -> None:
    assert plan(request_rate=4.0, duration_seconds=1.0, max_requests=20).arrival_offsets == (
        0.0,
        0.25,
        0.5,
        0.75,
    )
    assert plan(request_rate=100.0, duration_seconds=10.0, max_requests=3).arrival_offsets == (
        0.0,
        0.01,
        0.02,
    )


async def test_experiment_record_round_trips_configuration_summary_and_raw_observations(
    tmp_path: Path,
) -> None:
    async def succeed(call: int, texts: list[str]) -> GatewayCallResult:
        return GatewayCallResult(
            backend_id=f"backend-{call % 2}", vectors=len(texts), input_tokens=len(texts)
        )

    clock = ManualClock()
    load_plan = plan(max_requests=2)
    result = await RunGatewayLoad(
        ScriptedGatewayClient(succeed), clock=clock, sleep=clock.sleep
    ).run(load_plan)
    experiment = GatewayLoadExperiment(
        experiment_id="load123",
        created_at="2026-09-28T20:00:00Z",
        target=GatewayLoadTarget(
            base_url="http://127.0.0.1:8000",
            model="lab-embedding",
            label="local-two-simulator-gateway",
            kind="synthetic",
        ),
        client=GatewayLoadClientConfig(request_timeout_seconds=2.5),
        plan=load_plan,
        result=result,
    )

    store = GatewayLoadStore(tmp_path)
    path = store.save(experiment)
    loaded = store.load("load123")

    assert loaded == experiment
    assert path == tmp_path / "load123" / "gateway-load.json"
    observations = json.loads(
        (tmp_path / "load123" / "observations.json").read_text(encoding="utf-8")
    )
    assert len(observations) == 2
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["target"]["kind"] == "synthetic"
    assert saved["client"]["request_timeout_seconds"] == 2.5
    assert saved["plan"]["request_rate"] == 10.0
    assert saved["result"]["scheduled_requests"] == 2
    assert saved["result"]["backend_counts"] == {"backend-0": 1, "backend-1": 1}
    assert saved["result"]["observations"] == observations

    with pytest.raises(FileExistsError):
        store.save(experiment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "../escape"),
        ("created_at", ""),
    ],
)
def test_invalid_experiment_identity_is_not_persistable(field: str, value: str) -> None:
    values = {
        "experiment_id": "load123",
        "created_at": "2026-09-28T20:00:00Z",
        "target": GatewayLoadTarget(
            base_url="http://127.0.0.1:8000",
            model="lab-embedding",
            label="local",
            kind="synthetic",
        ),
        "client": GatewayLoadClientConfig(request_timeout_seconds=2.5),
        "plan": plan(max_requests=1),
        "result": None,
    }
    values[field] = value
    with pytest.raises(ValueError):
        GatewayLoadExperiment(**values)  # type: ignore[arg-type]
