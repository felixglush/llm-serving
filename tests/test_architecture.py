import ast
from pathlib import Path


def test_domain_does_not_import_infrastructure_packages() -> None:
    domain_directory = Path(__file__).parents[1] / "src" / "embedding_lab" / "domain"
    infrastructure_roots = {"boto3", "httpx", "modal", "requests", "vllm"}

    for source_path in domain_directory.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        imported_roots = {
            node.names[0].name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        }
        imported_roots.update(
            node.module.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert imported_roots.isdisjoint(infrastructure_roots), source_path
