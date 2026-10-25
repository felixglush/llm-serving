"""HTTPX adapter for the gateway's OpenAI-compatible embeddings endpoint."""

import math
from collections.abc import Mapping
from types import TracebackType

import httpx

from embedding_lab.adapters.openai_embeddings import (
    OpenAIEmbeddingResponseError,
    parse_embedding_response,
)
from embedding_lab.ports.gateway_client import (
    MAX_ERROR_LENGTH,
    GatewayCallResult,
    GatewayHttpError,
    GatewayProtocolError,
    GatewayRequestTimeout,
    GatewayTransportError,
)


class HttpxGatewayClient:
    """Send measured requests through one reusable asynchronous HTTP client."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        normalized_url = base_url.rstrip("/")
        endpoint = (
            f"{normalized_url}/embeddings"
            if normalized_url.endswith("/v1")
            else f"{normalized_url}/v1/embeddings"
        )
        parsed_endpoint = httpx.URL(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.host:
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        self._url = parsed_endpoint
        self._model = model.strip()
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def embed(self, texts: list[str]) -> GatewayCallResult:
        if not texts:
            raise ValueError("gateway embedding requests require at least one text")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("gateway embedding inputs must be strings")
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json={"model": self._model, "input": texts, "encoding_format": "float"},
            )
            if not response.is_success:
                error_code, message = _error_details(response)
                raise GatewayHttpError(response.status_code, error_code, message)
            try:
                payload = response.json()
            except ValueError as error:
                raise GatewayProtocolError("gateway returned invalid JSON") from error
            try:
                parsed = parse_embedding_response(
                    payload,
                    expected_count=len(texts),
                    expected_model=self._model,
                )
            except OpenAIEmbeddingResponseError as error:
                raise GatewayProtocolError(str(error)) from error
            backend_id = response.headers.get("X-Backend-ID", "").strip()
            if not backend_id:
                raise GatewayProtocolError("gateway success response did not include X-Backend-ID")
            return GatewayCallResult(
                backend_id=backend_id,
                vectors=len(parsed.embeddings),
                input_tokens=parsed.input_tokens,
            )
        except (GatewayHttpError, GatewayProtocolError):
            raise
        except httpx.TimeoutException as error:
            raise GatewayRequestTimeout("gateway request timed out") from error
        except httpx.RequestError as error:
            raise GatewayTransportError("could not reach the gateway") from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "HttpxGatewayClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _error_details(response: httpx.Response) -> tuple[str | None, str]:
    default = f"gateway returned HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        return None, default
    if not isinstance(payload, Mapping):
        return None, default
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return None, default
    code = error.get("code")
    message = error.get("message")
    bounded_code = code.strip()[:100] if isinstance(code, str) and code.strip() else None
    bounded_message = (
        message.strip()[:MAX_ERROR_LENGTH]
        if isinstance(message, str) and message.strip()
        else default
    )
    return bounded_code, bounded_message
