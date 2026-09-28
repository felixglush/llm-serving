"""Command-line composition root for Modal vLLM deployments."""

import argparse
import json
from collections.abc import Sequence

from vllm_model_server.domain import ModelConfig, ReplicaTopology
from vllm_model_server.modal_deployment import ModalDeploymentBackend, ModalDeploymentSettings


def main(argv: Sequence[str] | None = None) -> int:
    """Deploy one pinned vLLM embedding server through Modal."""

    parser = argparse.ArgumentParser(prog="vllm-model-server-deploy")
    parser.add_argument("--provider", choices=("modal",), default="modal")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision", help="immutable model revision")
    parser.add_argument("--max-sequence-length", type=int, default=8_192)
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-type", default="L4")
    parser.add_argument("--container-concurrency", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8_192)
    parser.add_argument("--min-containers", type=int, choices=(0, 1), default=0)
    parser.add_argument("--vllm-version", default="0.30.0")
    args = parser.parse_args(argv)

    try:
        settings = ModalDeploymentSettings(
            gpu_type=args.gpu_type,
            vllm_version=args.vllm_version,
            container_concurrency=args.container_concurrency,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            min_containers=args.min_containers,
        )
        model = ModelConfig(
            model_id=args.model,
            revision=args.revision,
            max_sequence_length=args.max_sequence_length,
            embedding_dimensions=args.embedding_dimensions,
        )
        topology = ReplicaTopology(
            replicas=args.replicas,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        deployment = ModalDeploymentBackend(settings).deploy(model, topology)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "name": deployment.name,
                "endpoint_url": deployment.endpoint_url,
                "configuration_fingerprint": deployment.configuration_fingerprint,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
