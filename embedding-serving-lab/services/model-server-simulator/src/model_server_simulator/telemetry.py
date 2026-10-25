"""Framework-neutral observability port for the simulator HTTP use case."""

from typing import Protocol


class SimulatorTelemetry(Protocol):
    def request_received(self) -> None: ...

    def request_completed(self, *, outcome: str, duration_seconds: float) -> None: ...

    def request_rejected(self, *, reason: str) -> None: ...

    def work_started(self) -> None: ...

    def work_completed(self) -> None: ...


class NoopSimulatorTelemetry:
    def request_received(self) -> None:
        return None

    def request_completed(self, *, outcome: str, duration_seconds: float) -> None:
        del outcome, duration_seconds

    def request_rejected(self, *, reason: str) -> None:
        del reason

    def work_started(self) -> None:
        return None

    def work_completed(self) -> None:
        return None
