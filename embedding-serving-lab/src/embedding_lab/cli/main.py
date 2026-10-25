"""Command-line entry point for embedding lab use cases."""

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import asdict

from embedding_lab.adapters.fake.embedding_service import FakeEmbeddingService
from embedding_lab.adapters.filesystem.gateway_load_store import GatewayLoadStore
from embedding_lab.adapters.gateway.http_client import HttpxGatewayClient
from embedding_lab.adapters.vllm.embedding_service import (
    VllmEmbeddingService,
    VllmRequestError,
)
from embedding_lab.adapters.vllm.schemas import VllmResponseError
from embedding_lab.adapters.wandb.publisher import WandbPublisher
from embedding_lab.application.benchmark import RunBenchmark
from embedding_lab.application.experiments import ExperimentStore, new_experiment_id, utc_now
from embedding_lab.application.gateway_load import RunGatewayLoad
from embedding_lab.application.verify import validate_embeddings
from embedding_lab.application.vllm_bench import (
    VllmBenchError,
    VllmBenchInvocation,
    VllmBenchStore,
    normalize_metrics,
)
from embedding_lab.domain.benchmark import BenchmarkResult
from embedding_lab.domain.deployment import Deployment
from embedding_lab.domain.embedding import EmbeddingResponse
from embedding_lab.domain.experiment import Experiment
from embedding_lab.domain.gateway_load import (
    FixedArrivalPlan,
    GatewayLoadClientConfig,
    GatewayLoadExperiment,
    GatewayLoadResult,
    GatewayLoadTarget,
)
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology
from embedding_lab.domain.vllm_benchmark import VllmBenchRecord
from embedding_lab.domain.workload import Workload

DEFAULT_TEXTS = ("A short example sentence.", "A different example sentence.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedding-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="check an embedding service response")
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
    experiment_parser = commands.add_parser("experiment", help="run and save a benchmark")
    experiment_parser.add_argument(
        "--service", choices=("fake", "vllm"), default="fake", help="service adapter to use"
    )
    experiment_parser.add_argument("--base-url", default=os.environ.get("EMBEDDING_BASE_URL"))
    experiment_parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
    )
    experiment_parser.add_argument("--revision")
    experiment_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    experiment_parser.add_argument("--concurrency", type=int, default=1)
    experiment_parser.add_argument("--requests", type=int, default=10)
    experiment_parser.add_argument("--texts-per-request", type=int, default=1)
    experiment_parser.add_argument("--target-tokens-per-text", type=int, default=32)
    experiment_parser.add_argument("--warmup-requests", type=int, default=1)
    experiment_parser.add_argument("--seed", type=int, default=0)
    experiment_parser.add_argument("--corpus-version", default="synthetic-v1")
    experiment_parser.add_argument("--deployment-name", default="local-embedding-service")
    experiment_parser.add_argument("--deployment-fingerprint", default="local")
    experiment_parser.add_argument("--endpoint-url")
    experiment_parser.add_argument("--gpu-type", default="unknown")
    experiment_parser.add_argument("--replicas", type=int, default=1)
    experiment_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    experiment_parser.add_argument("--results-dir", default="results")
    experiment_parser.add_argument("--publish", action="store_true")
    experiment_parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    experiment_parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    gateway_load_parser = commands.add_parser(
        "gateway-load",
        help="run and save a bounded fixed-arrival-rate gateway experiment",
    )
    gateway_load_parser.add_argument("--base-url", default=os.environ.get("GATEWAY_BASE_URL"))
    gateway_load_parser.add_argument(
        "--model",
        default=os.environ.get("GATEWAY_MODEL", "lab-embedding"),
    )
    gateway_load_parser.add_argument("--request-rate", type=float, default=10.0)
    gateway_load_parser.add_argument("--duration-seconds", type=float, default=10.0)
    gateway_load_parser.add_argument("--max-requests", type=int, default=100)
    gateway_load_parser.add_argument("--max-outstanding", type=int, default=32)
    gateway_load_parser.add_argument("--drain-timeout-seconds", type=float, default=5.0)
    gateway_load_parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    gateway_load_parser.add_argument("--texts-per-request", type=int, default=1)
    gateway_load_parser.add_argument("--target-tokens-per-text", type=int, default=32)
    gateway_load_parser.add_argument("--seed", type=int, default=0)
    gateway_load_parser.add_argument("--corpus-version", default="synthetic-v1")
    gateway_load_parser.add_argument("--target-label", default="gateway-load-target")
    gateway_load_parser.add_argument(
        "--target-kind",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    gateway_load_parser.add_argument("--results-dir", default="results/gateway-load")
    vllm_bench_parser = commands.add_parser(
        "vllm-bench", help="run vLLM's reference serving benchmark and optionally publish it"
    )
    vllm_bench_parser.add_argument("--base-url", default=os.environ.get("EMBEDDING_BASE_URL"))
    vllm_bench_parser.add_argument(
        "--model", default=os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    )
    vllm_bench_parser.add_argument("--revision", required=True)
    vllm_bench_parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", "vllm"))
    vllm_bench_parser.add_argument("--num-prompts", type=int, default=100)
    vllm_bench_parser.add_argument("--max-concurrency", type=int, default=1)
    vllm_bench_parser.add_argument("--input-len", type=int, default=32)
    vllm_bench_parser.add_argument("--num-warmups", type=int, default=1)
    vllm_bench_parser.add_argument("--request-rate", default="inf")
    vllm_bench_parser.add_argument("--ready-check-timeout-sec", type=int, default=900)
    vllm_bench_parser.add_argument("--seed", type=int, default=0)
    vllm_bench_parser.add_argument("--endpoint", default="/v1/embeddings")
    vllm_bench_parser.add_argument("--label", default="vllm-embeddings")
    vllm_bench_parser.add_argument("--deployment-name", default="modal-embedding-service")
    vllm_bench_parser.add_argument("--deployment-fingerprint", default="unknown")
    vllm_bench_parser.add_argument("--endpoint-url")
    vllm_bench_parser.add_argument("--gpu-type", default="unknown")
    vllm_bench_parser.add_argument("--replicas", type=int, default=1)
    vllm_bench_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    vllm_bench_parser.add_argument("--vllm-version", default="unknown")
    vllm_bench_parser.add_argument("--client-location", default="local")
    vllm_bench_parser.add_argument("--results-dir", default="results")
    vllm_bench_parser.add_argument("--publish", action="store_true")
    vllm_bench_parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    vllm_bench_parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
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
    if args.command == "experiment":
        if args.service == "vllm" and args.base_url is None:
            experiment_parser.error("--base-url or EMBEDDING_BASE_URL is required for vllm")
        try:
            workload = Workload(
                concurrency=args.concurrency,
                requests=args.requests,
                texts_per_request=args.texts_per_request,
                target_tokens_per_text=args.target_tokens_per_text,
                warmup_requests=args.warmup_requests,
                seed=args.seed,
                corpus_version=args.corpus_version,
            )
            topology = ReplicaTopology(
                replicas=args.replicas,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        except ValueError as error:
            experiment_parser.error(str(error))
        revision = args.revision or ("synthetic" if args.service == "fake" else None)
        if revision is None:
            experiment_parser.error("--revision is required for a persisted vllm experiment")
        benchmark_result = asyncio.run(_run_benchmark(args, workload))
        experiment = Experiment(
            experiment_id=new_experiment_id(),
            created_at=utc_now(),
            model=ModelConfig(model_id=args.model, revision=revision),
            topology=topology,
            deployment=Deployment(
                name=args.deployment_name,
                endpoint_url=args.endpoint_url or args.base_url or "local://embedding-service",
                configuration_fingerprint=args.deployment_fingerprint,
            ),
            serving_config={
                "service": args.service,
                "gpu_type": args.gpu_type,
                "timeout_seconds": args.timeout_seconds,
            },
            workload=workload,
            result=benchmark_result,
        )
        store = ExperimentStore(args.results_dir)
        experiment_path = store.save(experiment)
        output: dict[str, object] = {
            "experiment_id": experiment.experiment_id,
            "experiment_path": str(experiment_path),
            "experiment": experiment.to_dict(),
        }
        if args.publish:
            receipt = WandbPublisher(project=args.project, entity=args.entity).publish(
                experiment, store
            )
            output["wandb"] = asdict(receipt)
        print(json.dumps(output))
        return 0
    if args.command == "gateway-load":
        if args.base_url is None:
            gateway_load_parser.error("--base-url or GATEWAY_BASE_URL is required")
        try:
            load_plan = FixedArrivalPlan(
                request_rate=args.request_rate,
                duration_seconds=args.duration_seconds,
                max_requests=args.max_requests,
                max_outstanding=args.max_outstanding,
                drain_timeout_seconds=args.drain_timeout_seconds,
                texts_per_request=args.texts_per_request,
                target_tokens_per_text=args.target_tokens_per_text,
                seed=args.seed,
                corpus_version=args.corpus_version,
            )
            target = GatewayLoadTarget(
                base_url=args.base_url,
                model=args.model,
                label=args.target_label,
                kind=args.target_kind,
            )
            client_config = GatewayLoadClientConfig(
                request_timeout_seconds=args.request_timeout_seconds
            )
            gateway_load_result = asyncio.run(_run_gateway_load(target, load_plan, client_config))
            gateway_experiment = GatewayLoadExperiment(
                experiment_id=new_experiment_id(),
                created_at=utc_now(),
                target=target,
                client=client_config,
                plan=load_plan,
                result=gateway_load_result,
            )
            record_path = GatewayLoadStore(args.results_dir).save(gateway_experiment)
        except (ValueError, FileExistsError) as error:
            gateway_load_parser.error(str(error))
        summary = gateway_load_result.to_dict()
        summary.pop("observations")
        output = {
            "experiment_id": gateway_experiment.experiment_id,
            "record_path": str(record_path),
            "target": asdict(target),
            "client": asdict(client_config),
            "plan": asdict(load_plan),
            "result": summary,
            "capacity_claim": (
                "synthetic evidence; not production capacity"
                if target.kind == "synthetic"
                else "bounded environment-specific experiment"
            ),
        }
        print(json.dumps(output))
        return 0
    if args.command == "vllm-bench":
        if args.base_url is None:
            vllm_bench_parser.error("--base-url or EMBEDDING_BASE_URL is required")
        try:
            experiment_id = new_experiment_id()
            vllm_store = VllmBenchStore(args.results_dir)
            result_dir = vllm_store.directory(experiment_id)
            invocation = VllmBenchInvocation(
                executable=args.vllm_bin,
                base_url=args.base_url,
                model=args.model,
                result_dir=result_dir,
                num_prompts=args.num_prompts,
                max_concurrency=args.max_concurrency,
                input_len=args.input_len,
                num_warmups=args.num_warmups,
                request_rate=args.request_rate,
                seed=args.seed,
                endpoint=args.endpoint,
                ready_check_timeout_sec=args.ready_check_timeout_sec,
                label=args.label,
            )
            raw_result, command, raw_result_path = invocation.run()
            record = VllmBenchRecord(
                experiment_id=experiment_id,
                created_at=utc_now(),
                model=ModelConfig(model_id=args.model, revision=args.revision),
                topology=ReplicaTopology(
                    replicas=args.replicas,
                    tensor_parallel_size=args.tensor_parallel_size,
                ),
                deployment=Deployment(
                    name=args.deployment_name,
                    endpoint_url=args.endpoint_url or args.base_url,
                    configuration_fingerprint=args.deployment_fingerprint,
                ),
                serving_config={
                    "vllm_bin": args.vllm_bin,
                    "vllm_version": args.vllm_version,
                    "gpu_type": args.gpu_type,
                    "client_location": args.client_location,
                    "request_rate": args.request_rate,
                    "ready_check_timeout_sec": args.ready_check_timeout_sec,
                    "endpoint": args.endpoint,
                    "label": args.label,
                },
                workload={
                    "dataset_name": "random",
                    "num_prompts": args.num_prompts,
                    "max_concurrency": args.max_concurrency,
                    "input_len": args.input_len,
                    "num_warmups": args.num_warmups,
                    "seed": args.seed,
                },
                command=tuple(command),
                metrics=normalize_metrics(raw_result),
                raw_result=raw_result,
            )
            record_path = vllm_store.save(record)
            bench_output: dict[str, object] = {
                "experiment_id": experiment_id,
                "record_path": str(record_path),
                "raw_result_path": str(raw_result_path),
                "record": record.to_dict(),
            }
            if args.publish:
                receipt = WandbPublisher(
                    project=args.project, entity=args.entity
                ).publish_vllm_bench(record, vllm_store, str(raw_result_path))
                bench_output["wandb"] = asdict(receipt)
            print(json.dumps(bench_output))
        except (VllmBenchError, ValueError) as error:
            vllm_bench_parser.error(str(error))
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


async def _run_gateway_load(
    target: GatewayLoadTarget,
    plan: FixedArrivalPlan,
    client_config: GatewayLoadClientConfig,
) -> GatewayLoadResult:
    async with HttpxGatewayClient(
        base_url=target.base_url,
        model=target.model,
        api_key=os.environ.get("EMBEDDING_API_KEY"),
        timeout_seconds=client_config.request_timeout_seconds,
    ) as client:
        return await RunGatewayLoad(client).run(plan)


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
