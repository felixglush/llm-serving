"""Embedding service backed by vLLM's OpenAI-compatible HTTP API."""

import math
from collections.abc import Mapping
from types import TracebackType

import httpx

from embedding_lab.adapters.vllm.schemas import VllmResponseError, parse_embedding_response
from embedding_lab.domain.embedding import EmbeddingResponse


class VllmRequestError(RuntimeError):
    """A request to vLLM failed before a valid embedding response was available."""


MAX_ERROR_MESSAGE_LENGTH = 500


class VllmEmbeddingService:
    """Send embedding batches through one reusable asynchronous HTTP client."""

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

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        if not texts:
            raise ValueError("vLLM embedding requests require at least one text")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("vLLM embedding inputs must be strings")

        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                json={"model": self._model, "input": texts, "encoding_format": "float"},
            )
            response.raise_for_status()
            payload = response.json()
            return parse_embedding_response(
                payload,
                expected_count=len(texts),
                expected_model=self._model,
            )
        except VllmResponseError:
            raise
        except httpx.HTTPStatusError as error:
            detail = _structured_error_message(error.response)
            suffix = f": {detail}" if detail is not None else ""
            raise VllmRequestError(
                f"vLLM embedding request failed with HTTP {error.response.status_code}{suffix}"
            ) from error
        except httpx.RequestError as error:
            raise VllmRequestError("could not reach the vLLM embedding endpoint") from error
        except ValueError as error:
            raise VllmRequestError("vLLM returned invalid JSON") from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "VllmEmbeddingService":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _structured_error_message(response: httpx.Response) -> str | None:
    """Return a bounded message from vLLM's OpenAI error envelope, when present."""

    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return None
    message = error.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    return message.strip()[:MAX_ERROR_MESSAGE_LENGTH]
