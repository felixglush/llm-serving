"""Domain values for bounded fixed-arrival gateway load experiments."""

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from embedding_lab.domain.workload import synthetic_texts

LoadOutcome = Literal[
    "succeeded",
    "http_error",
    "timeout",
    "transport_error",
    "protocol_error",
    "client_error",
    "generator_dropped",
    "drain_timeout",
]
TargetKind = Literal["synthetic", "real"]


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FixedArrivalPlan:
    """A bounded arrival schedule independent of request completions."""

    request_rate: float
    duration_seconds: float
    max_requests: int
    max_outstanding: int
    drain_timeout_seconds: float
    texts_per_request: int
    target_tokens_per_text: int
    seed: int = 0
    corpus_version: str = "synthetic-v1"

    def __post_init__(self) -> None:
        _positive_finite("request_rate", self.request_rate)
        _positive_finite("duration_seconds", self.duration_seconds)
        _positive_integer("max_requests", self.max_requests)
        _positive_integer("max_outstanding", self.max_outstanding)
        _positive_finite("drain_timeout_seconds", self.drain_timeout_seconds)
        _positive_integer("texts_per_request", self.texts_per_request)
        _positive_integer("target_tokens_per_text", self.target_tokens_per_text)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.corpus_version.strip():
            raise ValueError("corpus_version must not be blank")

    @property
    def arrival_offsets(self) -> tuple[float, ...]:
        offsets: list[float] = []
        for request_index in range(self.max_requests):
            offset = request_index / self.request_rate
            if offset >= self.duration_seconds:
                break
            offsets.append(offset)
        return tuple(offsets)

    def texts_for_request(self, request_index: int) -> tuple[str, ...]:
        return synthetic_texts(
            seed=self.seed,
            corpus_version=self.corpus_version,
            request_index=request_index,
            texts_per_request=self.texts_per_request,
            target_tokens_per_text=self.target_tokens_per_text,
        )


@dataclass(frozen=True, slots=True)
class GatewayLoadTarget:
    """The externally deployed gateway measured by an experiment."""

    base_url: str
    model: str
    label: str
    kind: TargetKind

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.label.strip():
            raise ValueError("label must not be empty")
        if self.kind not in {"synthetic", "real"}:
            raise ValueError("kind must be synthetic or real")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


@dataclass(frozen=True, slots=True)
class GatewayLoadClientConfig:
    """Client behavior that materially changes experiment outcomes."""

    request_timeout_seconds: float

    def __post_init__(self) -> None:
        _positive_finite("request_timeout_seconds", self.request_timeout_seconds)


@dataclass(frozen=True, slots=True)
class GatewayLoadObservation:
    """One planned arrival and its observable terminal outcome."""

    request_index: int
    intended_dispatch_offset_seconds: float
    actual_dispatch_offset_seconds: float | None
    completion_offset_seconds: float | None
    dispatch_lag_ms: float | None
    latency_ms: float | None
    outcome: LoadOutcome
    status_code: int | None = None
    backend_id: str | None = None
    service_error_code: str | None = None
    error: str | None = None
    vectors: int = 0
    input_tokens: int = 0

    def __post_init__(self) -> None:
        if self.request_index < 0:
            raise ValueError("request_index must not be negative")
        _nonnegative_finite(
            "intended_dispatch_offset_seconds", self.intended_dispatch_offset_seconds
        )
        for name, value in (
            ("actual_dispatch_offset_seconds", self.actual_dispatch_offset_seconds),
            ("completion_offset_seconds", self.completion_offset_seconds),
            ("dispatch_lag_ms", self.dispatch_lag_ms),
            ("latency_ms", self.latency_ms),
        ):
            if value is not None:
                _nonnegative_finite(name, value)
        if self.outcome not in {
            "succeeded",
            "http_error",
            "timeout",
            "transport_error",
            "protocol_error",
            "client_error",
            "generator_dropped",
            "drain_timeout",
        }:
            raise ValueError("unsupported gateway load outcome")
        if self.outcome == "generator_dropped":
            if any(
                value is not None
                for value in (
                    self.actual_dispatch_offset_seconds,
                    self.completion_offset_seconds,
                    self.dispatch_lag_ms,
                    self.latency_ms,
                )
            ):
                raise ValueError("generator drops must not contain dispatch timing")
        elif any(
            value is None
            for value in (
                self.actual_dispatch_offset_seconds,
                self.completion_offset_seconds,
                self.dispatch_lag_ms,
                self.latency_ms,
            )
        ):
            raise ValueError("dispatched observations require complete timing")
        if self.outcome == "succeeded":
            if self.status_code != 200 or not self.backend_id:
                raise ValueError("successful observations require HTTP 200 and a backend")
            if self.error is not None or self.vectors < 1 or self.input_tokens < 1:
                raise ValueError("successful observations require validated response counts")
        elif not self.error:
            raise ValueError("unsuccessful observations require an error")
        if self.outcome == "http_error" and self.status_code is None:
            raise ValueError("HTTP error observations require a status code")


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GatewayLoadResult:
    """Raw fixed-arrival observations plus reproducible derived summaries."""

    target_request_rate: float
    arrival_window_seconds: float
    arrival_phase_elapsed_seconds: float
    total_elapsed_seconds: float
    max_observed_outstanding: int
    observations: tuple[GatewayLoadObservation, ...]

    def __post_init__(self) -> None:
        _positive_finite("target_request_rate", self.target_request_rate)
        _positive_finite("arrival_window_seconds", self.arrival_window_seconds)
        _nonnegative_finite("arrival_phase_elapsed_seconds", self.arrival_phase_elapsed_seconds)
        _positive_finite("total_elapsed_seconds", self.total_elapsed_seconds)
        if self.arrival_phase_elapsed_seconds > self.total_elapsed_seconds:
            raise ValueError("arrival phase must not exceed total elapsed time")
        if self.max_observed_outstanding < 0:
            raise ValueError("max_observed_outstanding must not be negative")
        indices = [item.request_index for item in self.observations]
        if indices != list(range(len(self.observations))):
            raise ValueError("observations must contain every planned request in order")

    @property
    def scheduled_requests(self) -> int:
        return len(self.observations)

    @property
    def dispatched_requests(self) -> int:
        return sum(item.actual_dispatch_offset_seconds is not None for item in self.observations)

    @property
    def in_window_dispatched_requests(self) -> int:
        return sum(
            item.actual_dispatch_offset_seconds is not None
            and item.actual_dispatch_offset_seconds <= self.arrival_window_seconds
            for item in self.observations
        )

    @property
    def dispatch_shortfall_requests(self) -> int:
        return self.scheduled_requests - self.in_window_dispatched_requests

    @property
    def successful_requests(self) -> int:
        return sum(item.outcome == "succeeded" for item in self.observations)

    @property
    def failed_requests(self) -> int:
        return self.dispatched_requests - self.successful_requests

    @property
    def generator_dropped_requests(self) -> int:
        return sum(item.outcome == "generator_dropped" for item in self.observations)

    @property
    def drain_timed_out_requests(self) -> int:
        return sum(item.outcome == "drain_timeout" for item in self.observations)

    @property
    def outcome_counts(self) -> dict[str, int]:
        return dict(Counter(item.outcome for item in self.observations))

    @property
    def status_code_counts(self) -> dict[int, int]:
        return dict(
            Counter(item.status_code for item in self.observations if item.status_code is not None)
        )

    @property
    def backend_counts(self) -> dict[str, int]:
        return dict(
            Counter(item.backend_id for item in self.observations if item.backend_id is not None)
        )

    @property
    def max_dispatch_lag_ms(self) -> float | None:
        lags = [
            item.dispatch_lag_ms for item in self.observations if item.dispatch_lag_ms is not None
        ]
        return max(lags) if lags else None

    @property
    def p50_latency_ms(self) -> float | None:
        return _percentile(self._successful_latencies, 0.50)

    @property
    def p95_latency_ms(self) -> float | None:
        return _percentile(self._successful_latencies, 0.95)

    @property
    def p99_latency_ms(self) -> float | None:
        return _percentile(self._successful_latencies, 0.99)

    @property
    def actual_dispatch_rate(self) -> float:
        return self.in_window_dispatched_requests / self.arrival_window_seconds

    @property
    def successful_completion_rate_during_arrival_window(self) -> float:
        successes = sum(
            item.outcome == "succeeded"
            and item.completion_offset_seconds is not None
            and item.completion_offset_seconds <= self.arrival_window_seconds
            for item in self.observations
        )
        return successes / self.arrival_window_seconds

    @property
    def whole_run_successful_requests_per_second(self) -> float:
        return self.successful_requests / self.total_elapsed_seconds

    @property
    def _successful_latencies(self) -> tuple[float, ...]:
        return tuple(
            item.latency_ms
            for item in self.observations
            if item.outcome == "succeeded" and item.latency_ms is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_request_rate": self.target_request_rate,
            "arrival_window_seconds": self.arrival_window_seconds,
            "arrival_phase_elapsed_seconds": self.arrival_phase_elapsed_seconds,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "max_observed_outstanding": self.max_observed_outstanding,
            "scheduled_requests": self.scheduled_requests,
            "dispatched_requests": self.dispatched_requests,
            "in_window_dispatched_requests": self.in_window_dispatched_requests,
            "dispatch_shortfall_requests": self.dispatch_shortfall_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "generator_dropped_requests": self.generator_dropped_requests,
            "drain_timed_out_requests": self.drain_timed_out_requests,
            "outcome_counts": self.outcome_counts,
            "status_code_counts": self.status_code_counts,
            "backend_counts": self.backend_counts,
            "max_dispatch_lag_ms": self.max_dispatch_lag_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "actual_dispatch_rate": self.actual_dispatch_rate,
            "successful_completion_rate_during_arrival_window": (
                self.successful_completion_rate_during_arrival_window
            ),
            "whole_run_successful_requests_per_second": (
                self.whole_run_successful_requests_per_second
            ),
            "observations": [asdict(item) for item in self.observations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GatewayLoadResult":
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise ValueError("result.observations must be a list")
        return cls(
            target_request_rate=_number(payload, "target_request_rate"),
            arrival_window_seconds=_number(payload, "arrival_window_seconds"),
            arrival_phase_elapsed_seconds=_number(payload, "arrival_phase_elapsed_seconds"),
            total_elapsed_seconds=_number(payload, "total_elapsed_seconds"),
            max_observed_outstanding=_integer(payload, "max_observed_outstanding"),
            observations=tuple(
                GatewayLoadObservation(**_mapping_value(item)) for item in observations
            ),
        )


@dataclass(frozen=True, slots=True)
class GatewayLoadExperiment:
    """A versioned, locally persisted gateway load experiment."""

    experiment_id: str
    created_at: str
    target: GatewayLoadTarget
    client: GatewayLoadClientConfig
    plan: FixedArrivalPlan
    result: GatewayLoadResult
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if "/" in self.experiment_id or "\\" in self.experiment_id or ".." in self.experiment_id:
            raise ValueError("experiment_id must be a path-safe identifier")
        if not self.created_at.strip():
            raise ValueError("created_at must not be empty")
        if self.schema_version != 1:
            raise ValueError("unsupported gateway load schema version")
        if len(self.plan.arrival_offsets) != self.result.scheduled_requests:
            raise ValueError("result must account for every planned arrival")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "target": asdict(self.target),
            "client": asdict(self.client),
            "plan": asdict(self.plan),
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GatewayLoadExperiment":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported gateway load schema version")
        return cls(
            experiment_id=_string(payload, "experiment_id"),
            created_at=_string(payload, "created_at"),
            target=GatewayLoadTarget(**_mapping(payload, "target")),
            client=GatewayLoadClientConfig(**_mapping(payload, "client")),
            plan=FixedArrivalPlan(**_mapping(payload, "plan")),
            result=GatewayLoadResult.from_dict(_mapping(payload, "result")),
            schema_version=1,
        )


def _percentile(values: tuple[float, ...], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _mapping_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("observation must be an object")
    return dict(value)


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value
