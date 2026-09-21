"""Command-line entry point for embedding lab use cases."""

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import asdict

from embedding_lab.adapters.fake.embedding_service import FakeEmbeddingService
from embedding_lab.adapters.modal.deployment import (
    ModalDeploymentBackend,
    ModalDeploymentSettings,
)
from embedding_lab.adapters.vllm.embedding_service import (
    VllmEmbeddingService,
    VllmRequestError,
)
from embedding_lab.adapters.vllm.schemas import VllmResponseError
from embedding_lab.application.benchmark import RunBenchmark
from embedding_lab.application.verify import validate_embeddings
from embedding_lab.domain.benchmark import BenchmarkResult
from embedding_lab.domain.embedding import EmbeddingResponse
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology
from embedding_lab.domain.workload import Workload

DEFAULT_TEXTS = ("A short example sentence.", "A different example sentence.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedding-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser(
        "verify", help="check an embedding service response"
    )
    verify_parser.add_argument(
        "--service", choices=("fake", "vllm"), default="fake", help="service adapter to use"
    )
    verify_parser.add_argument("--text", action="append", help="text to embed; may be repeated")
    verify_parser.add_argument("--expected-dimensions", type=int)
    verify_parser.add_argument(
        "--base-url",
        default=os.environ.get("EMBEDDING_BASE_URL"),
        help="vLLM server URL (or EMBEDDING_BASE_URL)",
    )
    verify_parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        help="served model name (or EMBEDDING_MODEL)",
    )
    verify_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    benchmark_parser = commands.add_parser("benchmark", help="measure an embedding service")
    benchmark_parser.add_argument(
        "--service", choices=("fake", "vllm"), default="fake", help="service adapter to use"
    )
    benchmark_parser.add_argument("--base-url", default=os.environ.get("EMBEDDING_BASE_URL"))
    benchmark_parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    )
    benchmark_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    benchmark_parser.add_argument("--concurrency", type=int, default=1)
    benchmark_parser.add_argument("--requests", type=int, default=10)
    benchmark_parser.add_argument("--texts-per-request", type=int, default=1)
    benchmark_parser.add_argument("--target-tokens-per-text", type=int, default=32)
    benchmark_parser.add_argument("--warmup-requests", type=int, default=1)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--corpus-version", default="synthetic-v1")
    deploy_parser = commands.add_parser("deploy", help="deploy an embedding service")
    deploy_parser.add_argument("--provider", choices=("modal",), required=True)
    deploy_parser.add_argument(
        "--model", default="Qwen/Qwen3-Embedding-0.6B", help="model repository ID"
    )
    deploy_parser.add_argument("--revision", help="immutable model revision")
    deploy_parser.add_argument("--max-sequence-length", type=int, default=8_192)
    deploy_parser.add_argument("--embedding-dimensions", type=int)
    deploy_parser.add_argument("--replicas", type=int, default=1)
    deploy_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    deploy_parser.add_argument("--gpu-type", default="L4")
    deploy_parser.add_argument("--container-concurrency", type=int, default=32)
    deploy_parser.add_argument("--max-num-seqs", type=int, default=32)
    deploy_parser.add_argument("--max-num-batched-tokens", type=int, default=8_192)
    deploy_parser.add_argument("--min-containers", type=int, choices=(0, 1), default=0)
    deploy_parser.add_argument("--vllm-version", default="0.29.0")
    args = parser.parse_args(argv)

    if args.command == "verify":
        if args.expected_dimensions is not None and args.expected_dimensions < 1:
            verify_parser.error("--expected-dimensions must be positive")
        if args.service == "vllm" and args.base_url is None:
            verify_parser.error("--base-url or EMBEDDING_BASE_URL is required for vllm")
        texts = args.text if args.text is not None else DEFAULT_TEXTS
        try:
            response = asyncio.run(_embed(args, texts))
        except (VllmRequestError, VllmResponseError, ValueError) as error:
            _print_result(args.service, len(texts), None, (str(error),))
            return 1
        result = validate_embeddings(
            texts,
            response.embeddings,
            expected_dimensions=args.expected_dimensions,
        )
        _print_result(
            args.service,
            result.input_count,
            result.dimensions,
            result.failures,
        )
        return 0 if result.passed else 1
    if args.command == "benchmark":
        if args.service == "vllm" and args.base_url is None:
            benchmark_parser.error("--base-url or EMBEDDING_BASE_URL is required for vllm")
        workload = Workload(
            concurrency=args.concurrency,
            requests=args.requests,
            texts_per_request=args.texts_per_request,
            target_tokens_per_text=args.target_tokens_per_text,
            warmup_requests=args.warmup_requests,
            seed=args.seed,
            corpus_version=args.corpus_version,
        )
        benchmark_result = asyncio.run(_run_benchmark(args, workload))
        result_payload = asdict(benchmark_result)
        result_payload.update(
            {
                "failure_rate": benchmark_result.failure_rate,
                "requests_per_second": benchmark_result.requests_per_second,
                "tokens_per_second": benchmark_result.tokens_per_second,
            }
        )
        print(
            json.dumps(
                {
                    "service": args.service,
                    "workload": asdict(workload),
                    "result": result_payload,
                }
            )
        )
        return 0
    if args.command == "deploy":
        settings = ModalDeploymentSettings(
            gpu_type=args.gpu_type,
            vllm_version=args.vllm_version,
            container_concurrency=args.container_concurrency,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            min_containers=args.min_containers,
        )
        deployment = ModalDeploymentBackend(settings).deploy(
            ModelConfig(
                model_id=args.model,
                revision=args.revision,
                max_sequence_length=args.max_sequence_length,
                embedding_dimensions=args.embedding_dimensions,
            ),
            ReplicaTopology(
                replicas=args.replicas,
                tensor_parallel_size=args.tensor_parallel_size,
            ),
        )
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
    return 2


async def _embed(args: argparse.Namespace, texts: Sequence[str]) -> EmbeddingResponse:
    if args.service == "fake":
        return await FakeEmbeddingService().embed(list(texts))

    async with VllmEmbeddingService(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get("EMBEDDING_API_KEY"),
        timeout_seconds=args.timeout_seconds,
    ) as service:
        return await service.embed(list(texts))


async def _run_benchmark(args: argparse.Namespace, workload: Workload) -> BenchmarkResult:
    if args.service == "fake":
        return await RunBenchmark(FakeEmbeddingService()).run(workload)

    async with VllmEmbeddingService(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get("EMBEDDING_API_KEY"),
        timeout_seconds=args.timeout_seconds,
    ) as service:
        return await RunBenchmark(service).run(workload)


def _print_result(
    service: str,
    input_count: int,
    dimensions: int | None,
    failures: Sequence[str],
) -> None:
    print(
        json.dumps(
            {
                "service": service,
                "passed": not failures,
                "input_count": input_count,
                "dimensions": dimensions,
                "failures": failures,
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
