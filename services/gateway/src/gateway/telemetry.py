"""Framework-neutral observability port for the gateway use cases."""

from typing import Protocol

from gateway.domain import BackendId


class GatewayTelemetry(Protocol):
    """Record bounded events without exposing a metrics implementation."""

    def request_received(self, *, route: str) -> None: ...

    def request_completed(self, *, route: str, outcome: str, duration_seconds: float) -> None: ...

    def request_rejected(self, *, route: str, reason: str) -> None: ...

    def backend_selected(self, *, backend_id: BackendId) -> None: ...

    def upstream_started(self, *, backend_id: BackendId) -> None: ...

    def upstream_completed(
        self,
        *,
        backend_id: BackendId,
        outcome: str,
        duration_seconds: float,
    ) -> None: ...


class NoopGatewayTelemetry:
    """Default adapter used when a caller does not configure observability."""

    def request_received(self, *, route: str) -> None:
        del route

    def request_completed(self, *, route: str, outcome: str, duration_seconds: float) -> None:
        del route, outcome, duration_seconds

    def request_rejected(self, *, route: str, reason: str) -> None:
        del route, reason

    def backend_selected(self, *, backend_id: BackendId) -> None:
        del backend_id

    def upstream_started(self, *, backend_id: BackendId) -> None:
        del backend_id

    def upstream_completed(
        self,
        *,
        backend_id: BackendId,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        del backend_id, outcome, duration_seconds
