import pytest

from model_server.domain import ReplicaTopology
from model_server.runtime import VllmEngineConfig


def test_vllm_config_translates_topology_into_engine_arguments() -> None:
    config = VllmEngineConfig.from_topology(
        ReplicaTopology(replicas=1, tensor_parallel_size=2),
        max_model_len=8_192,
        max_num_seqs=32,
        max_num_batched_tokens=8_192,
    )

    assert config.to_cli_args() == (
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "8192",
        "--tensor-parallel-size",
        "2",
    )


def test_one_server_command_does_not_silently_discard_replica_count() -> None:
    with pytest.raises(ValueError, match="exactly one replica"):
        VllmEngineConfig.from_topology(
            ReplicaTopology(replicas=2, tensor_parallel_size=1),
            max_model_len=8_192,
            max_num_seqs=32,
            max_num_batched_tokens=8_192,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_len", 0),
        ("max_num_seqs", -1),
        ("max_num_batched_tokens", True),
        ("tensor_parallel_size", 0),
    ],
)
def test_vllm_config_rejects_invalid_engine_limits(field: str, value: object) -> None:
    values: dict[str, object] = {
        "max_model_len": 8_192,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 8_192,
        "tensor_parallel_size": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        VllmEngineConfig(**values)  # type: ignore[arg-type]
