"""Configuration values owned by the vLLM model-server service."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A model and the limits this service will provision."""

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


@dataclass(frozen=True, slots=True)
class ReplicaTopology:
    """How workers form the one deployment managed by this service."""

    replicas: int = 1
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.replicas < 1:
            raise ValueError("replicas must be positive")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")


@dataclass(frozen=True, slots=True)
class Deployment:
    """The endpoint created for one vLLM serving configuration."""

    name: str
    endpoint_url: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("name", "endpoint_url", "configuration_fingerprint"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
