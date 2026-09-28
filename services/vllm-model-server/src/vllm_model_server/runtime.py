"""Translate portable replica topology into vLLM server arguments."""

from dataclasses import dataclass

from vllm_model_server.domain import ReplicaTopology


@dataclass(frozen=True, slots=True)
class VllmEngineConfig:
    """Tuning knobs owned by the vLLM infrastructure boundary.

    The domain describes how workers form replicas. This adapter-specific object
    describes how one vLLM process should schedule work inside a replica.
    """

    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        for name in (
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "tensor_parallel_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_topology(
        cls,
        topology: ReplicaTopology,
        *,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
    ) -> "VllmEngineConfig":
        """Configure one local vLLM server from a provider-neutral topology."""

        if topology.replicas != 1:
            raise ValueError(
                "one vLLM server command configures exactly one replica; "
                "replica orchestration belongs to a deployment backend"
            )
        return cls(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            tensor_parallel_size=topology.tensor_parallel_size,
        )

    def to_cli_args(self) -> tuple[str, ...]:
        """Render only the engine flags owned by this configuration."""

        return (
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        )
