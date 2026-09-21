"""Deserialize the OpenAI-compatible embedding response returned by vLLM."""

from collections.abc import Mapping

from embedding_lab.domain.embedding import Embedding, EmbeddingResponse


class VllmResponseError(ValueError):
    """The vLLM response did not satisfy the embeddings API contract."""


def parse_embedding_response(payload: object, expected_count: int) -> EmbeddingResponse:
    if not isinstance(payload, Mapping):
        raise VllmResponseError("response must be a JSON object")

    raw_data = payload.get("data")
    raw_usage = payload.get("usage")
    if not isinstance(raw_data, list):
        raise VllmResponseError("response.data must be a list")
    if not isinstance(raw_usage, Mapping):
        raise VllmResponseError("response.usage must be an object")

    prompt_tokens = raw_usage.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise VllmResponseError("response.usage.prompt_tokens must be an integer")
    if prompt_tokens < 1:
        raise VllmResponseError("response.usage.prompt_tokens must be positive")

    by_index: dict[int, Embedding] = {}
    for item in raw_data:
        index, embedding = _parse_item(item)
        if index in by_index:
            raise VllmResponseError(f"response.data contains duplicate index {index}")
        by_index[index] = embedding

    expected_indices = set(range(expected_count))
    if set(by_index) != expected_indices:
        raise VllmResponseError(
            f"response.data indices must be 0 through {expected_count - 1}"
        )

    return EmbeddingResponse(
        embeddings=tuple(by_index[index] for index in range(expected_count)),
        input_tokens=prompt_tokens,
    )


def _parse_item(item: object) -> tuple[int, Embedding]:
    if not isinstance(item, Mapping):
        raise VllmResponseError("each response.data item must be an object")

    index = item.get("index")
    raw_embedding = item.get("embedding")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise VllmResponseError("response.data indices must be non-negative integers")
    if not isinstance(raw_embedding, list):
        raise VllmResponseError(f"response.data[{index}].embedding must be a list")

    values: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VllmResponseError(
                f"response.data[{index}].embedding must contain only numbers"
            )
        try:
            values.append(float(value))
        except OverflowError as error:
            raise VllmResponseError(
                f"response.data[{index}].embedding contains an out-of-range number"
            ) from error
    return index, Embedding(values=tuple(values))
