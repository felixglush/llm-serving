"""HTTPX adapter for one configured embedding backend."""

import math
from collections.abc import Mapping

import httpx

from gateway.domain import (
    BackendId,
    EmbedCommand,
    Embedding,
    EmbeddingResponse,
    RequestContext,
)
from gateway.failures import InvalidUpstreamResponse, UpstreamUnavailable


class HttpxBackendClient:
    """Reuse one asynchronous HTTP client and one credential per backend."""

    def __init__(
        self,
        *,
        backend_id: BackendId,
        base_url: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        endpoint = (
            f"{normalized_url}/embeddings"
            if normalized_url.endswith("/v1")
            else f"{normalized_url}/v1/embeddings"
        )
        parsed_endpoint = httpx.URL(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.host:
            raise ValueError("backend base URL must be an HTTP or HTTPS URL")
        self._backend_id = backend_id
        self._endpoint = parsed_endpoint
        health_base = normalized_url[:-3] if normalized_url.endswith("/v1") else normalized_url
        self._health_url = httpx.URL(f"{health_base}/health")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=30.0, write=2.0, pool=2.0)
        )

    @property
    def backend_id(self) -> BackendId:
        return self._backend_id

    async def embed(self, command: EmbedCommand, context: RequestContext) -> EmbeddingResponse:
        headers = {**self._headers, "X-Request-ID": context.request_id}
        try:
            response = await self._client.post(
                self._endpoint,
                headers=headers,
                json={
                    "model": command.model,
                    "input": command.inputs,
                    "encoding_format": command.encoding_format,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise UpstreamUnavailable from error

        try:
            payload = response.json()
        except ValueError as error:
            raise InvalidUpstreamResponse from error
        return _parse_embedding_response(payload)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get(
                self._health_url,
                headers=self._headers,
                timeout=1.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False


def _parse_embedding_response(payload: object) -> EmbeddingResponse:
    if not isinstance(payload, Mapping):
        raise InvalidUpstreamResponse
    model = payload.get("model")
    data = payload.get("data")
    usage = payload.get("usage")
    if (
        payload.get("object") != "list"
        or not isinstance(model, str)
        or not isinstance(data, list)
        or not isinstance(usage, Mapping)
    ):
        raise InvalidUpstreamResponse

    embeddings_by_index: dict[int, Embedding] = {}
    for raw_item in data:
        if not isinstance(raw_item, Mapping):
            raise InvalidUpstreamResponse
        index = raw_item.get("index")
        raw_values = raw_item.get("embedding")
        if (
            raw_item.get("object") != "embedding"
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in embeddings_by_index
            or not isinstance(raw_values, list)
        ):
            raise InvalidUpstreamResponse
        values: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidUpstreamResponse
            try:
                numeric_value = float(value)
            except OverflowError as error:
                raise InvalidUpstreamResponse from error
            if not math.isfinite(numeric_value):
                raise InvalidUpstreamResponse
            values.append(numeric_value)
        embeddings_by_index[index] = Embedding(index=index, values=tuple(values))

    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or prompt_tokens < 0
        or total_tokens < 0
    ):
        raise InvalidUpstreamResponse
    return EmbeddingResponse(
        model=model,
        embeddings=tuple(embeddings_by_index[index] for index in sorted(embeddings_by_index)),
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )
