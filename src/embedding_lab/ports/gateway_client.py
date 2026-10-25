"""Gateway HTTP boundary consumed by fixed-arrival load experiments."""

from dataclasses import dataclass
from typing import Protocol

MAX_ERROR_LENGTH = 500


class GatewayClientError(RuntimeError):
    """A bounded client-side failure for one gateway request."""

    def __init__(self, message: str) -> None:
        bounded = message.strip()[:MAX_ERROR_LENGTH] or type(self).__name__
        super().__init__(bounded)


class GatewayHttpError(GatewayClientError):
    """A non-successful HTTP response returned by the gateway."""

    def __init__(self, status_code: int, error_code: str | None, message: str) -> None:
        if status_code < 300 or status_code > 599:
            raise ValueError("status_code must be a non-successful HTTP status")
        self.status_code = status_code
        self.error_code = error_code.strip()[:100] if error_code else None
        super().__init__(message)


class GatewayRequestTimeout(GatewayClientError):
    """The load client deadline expired before a response arrived."""


class GatewayTransportError(GatewayClientError):
    """The gateway could not be reached through the HTTP transport."""


class GatewayProtocolError(GatewayClientError):
    """A successful HTTP response violated the embeddings wire contract."""


@dataclass(frozen=True, slots=True)
class GatewayCallResult:
    """Validated data retained from one successful gateway response."""

    backend_id: str
    vectors: int
    input_tokens: int

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must not be empty")
        if self.vectors < 1:
            raise ValueError("vectors must be positive")
        if self.input_tokens < 1:
            raise ValueError("input_tokens must be positive")


class GatewayClient(Protocol):
    """Send one embedding request through the gateway's public HTTP contract."""

    async def embed(self, texts: list[str]) -> GatewayCallResult: ...
