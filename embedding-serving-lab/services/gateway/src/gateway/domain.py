"""Gateway-owned values that do not depend on HTTP frameworks or clients."""

from dataclasses import dataclass
from typing import Literal, NewType

BackendId = NewType("BackendId", str)


@dataclass(frozen=True, slots=True)
class EmbedCommand:
    """One validated embedding request for the configured model."""

    model: str
    inputs: tuple[str, ...]
    encoding_format: Literal["float"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Gateway-generated identity and absolute event-loop deadline."""

    request_id: str
    deadline: float


@dataclass(frozen=True, slots=True)
class Embedding:
    """One indexed vector returned by a backend."""

    index: int
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """A backend response translated into gateway-owned values."""

    model: str
    embeddings: tuple[Embedding, ...]
    prompt_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class RoutedEmbeddingResult:
    """A successful response together with the configured backend identity."""

    backend_id: BackendId
    response: EmbeddingResponse
