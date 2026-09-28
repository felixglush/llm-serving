"""Typed application failures that the inbound adapter maps to HTTP."""


class GatewayFailure(RuntimeError):
    """A safe, expected failure while handling an embedding request."""

    code = "gateway_failure"
    safe_message = "gateway request failed"
    retryable = False

    def __init__(
        self,
        safe_message: str | None = None,
        *,
        backend_id: str | None = None,
    ) -> None:
        if safe_message is not None:
            self.safe_message = safe_message[:500]
        self.backend_id = backend_id
        super().__init__(self.safe_message)


class InvalidRequest(GatewayFailure):
    code = "invalid_request"
    safe_message = "request is invalid"


class RequestTooLarge(GatewayFailure):
    code = "request_too_large"
    safe_message = "request body is too large"


class UnknownModel(GatewayFailure):
    code = "unknown_model"
    safe_message = "model must match the configured model"


class NoUsableBackend(GatewayFailure):
    code = "no_usable_backend"
    safe_message = "no backend is available"
    retryable = True


class UpstreamUnavailable(GatewayFailure):
    code = "upstream_unavailable"
    safe_message = "selected backend is unavailable"
    retryable = True


class InvalidUpstreamResponse(GatewayFailure):
    code = "invalid_upstream_response"
    safe_message = "selected backend returned an invalid response"


class DeadlineExceeded(GatewayFailure):
    code = "deadline_exceeded"
    safe_message = "gateway request deadline exceeded"
    retryable = True
