"""Deserialize the shared OpenAI-compatible embeddings wire response."""

import math
from collections.abc import Mapping

from embedding_lab.domain.embedding import Embedding, EmbeddingResponse


class OpenAIEmbeddingResponseError(ValueError):
    """The response did not satisfy the supported embeddings wire contract."""


def parse_embedding_response(
    payload: object, expected_count: int, *, expected_model: str | None = None
) -> EmbeddingResponse:
    if not isinstance(payload, Mapping):
        raise OpenAIEmbeddingResponseError("response must be a JSON object")

    raw_data = payload.get("data")
    raw_usage = payload.get("usage")
    if not isinstance(raw_data, list):
        raise OpenAIEmbeddingResponseError("response.data must be a list")
    if not isinstance(raw_usage, Mapping):
        raise OpenAIEmbeddingResponseError("response.usage must be an object")

    prompt_tokens = raw_usage.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise OpenAIEmbeddingResponseError("response.usage.prompt_tokens must be an integer")
    if prompt_tokens < 1:
        raise OpenAIEmbeddingResponseError("response.usage.prompt_tokens must be positive")

    by_index: dict[int, Embedding] = {}
    for item in raw_data:
        index, embedding = _parse_item(item)
        if index in by_index:
            raise OpenAIEmbeddingResponseError(f"response.data contains duplicate index {index}")
        by_index[index] = embedding

    expected_indices = set(range(expected_count))
    if set(by_index) != expected_indices:
        raise OpenAIEmbeddingResponseError(
            f"response.data indices must be 0 through {expected_count - 1}"
        )

    if payload.get("object") != "list":
        raise OpenAIEmbeddingResponseError("response.object must be list")
    response_model = payload.get("model")
    if not isinstance(response_model, str) or not response_model.strip():
        raise OpenAIEmbeddingResponseError("response.model must be a non-empty string")
    if expected_model is not None and response_model != expected_model:
        raise OpenAIEmbeddingResponseError("response.model must match the requested model")

    return EmbeddingResponse(
        embeddings=tuple(by_index[index] for index in range(expected_count)),
        input_tokens=prompt_tokens,
    )


def _parse_item(item: object) -> tuple[int, Embedding]:
    if not isinstance(item, Mapping):
        raise OpenAIEmbeddingResponseError("each response.data item must be an object")
    if item.get("object") != "embedding":
        raise OpenAIEmbeddingResponseError("each response.data item object must be embedding")

    index = item.get("index")
    raw_embedding = item.get("embedding")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise OpenAIEmbeddingResponseError("response.data indices must be non-negative integers")
    if not isinstance(raw_embedding, list):
        raise OpenAIEmbeddingResponseError(f"response.data[{index}].embedding must be a list")

    values: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OpenAIEmbeddingResponseError(
                f"response.data[{index}].embedding must contain only numbers"
            )
        try:
            parsed_value = float(value)
        except OverflowError as error:
            raise OpenAIEmbeddingResponseError(
                f"response.data[{index}].embedding contains an out-of-range number"
            ) from error
        if not math.isfinite(parsed_value):
            raise OpenAIEmbeddingResponseError(
                f"response.data[{index}].embedding must contain only finite numbers"
            )
        values.append(parsed_value)
    return index, Embedding(values=tuple(values))
