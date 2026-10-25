"""Model identity and limits used by research experiments."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A model and the limits the experiment expects it to honor."""

    model_id: str
    revision: str | None = None
    max_sequence_length: int = 2_048
    embedding_dimensions: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must not be blank when provided")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if self.embedding_dimensions is not None and self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be positive when provided")
