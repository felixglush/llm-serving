"""Vendor-neutral concepts used by the embedding lab."""

from embedding_lab.domain.benchmark import BenchmarkResult
from embedding_lab.domain.model import ModelConfig
from embedding_lab.domain.topology import ReplicaTopology
from embedding_lab.domain.workload import Workload

__all__ = ["BenchmarkResult", "ModelConfig", "ReplicaTopology", "Workload"]
