"""Protocols implemented by infrastructure adapters."""

from embedding_lab.ports.deployment_backend import DeploymentBackend
from embedding_lab.ports.embedding_service import EmbeddingService

__all__ = ["DeploymentBackend", "EmbeddingService"]
