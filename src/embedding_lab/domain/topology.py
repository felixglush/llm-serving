"""Replica topology concepts shared by deployment adapters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReplicaTopology:
    """How many replicas run and how many workers share each replica."""

    replicas: int = 1
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.replicas < 1:
            raise ValueError("replicas must be positive")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")

    @property
    def worker_count(self) -> int:
        """Return the total number of workers implied by this topology."""

        return self.replicas * self.tensor_parallel_size
