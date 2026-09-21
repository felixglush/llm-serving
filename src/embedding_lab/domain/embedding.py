"""Provider-neutral embedding responses with request-level input-token usage."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Embedding:
    """One vector in input order; correctness is checked by the verification use case."""

    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Vectors and the service's input-token count for the entire request."""

    embeddings: tuple[Embedding, ...]
    input_tokens: int

    def __post_init__(self) -> None:
        if isinstance(self.input_tokens, bool) or not isinstance(self.input_tokens, int):
            raise ValueError("input_tokens must be an integer reported by the service")
        if self.input_tokens < 0:
            raise ValueError("input_tokens must not be negative")
        if self.embeddings and self.input_tokens == 0:
            raise ValueError("embeddings require a positive input-token count")
