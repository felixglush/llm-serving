import ast
from pathlib import Path


def test_domain_does_not_import_outward_layers_or_infrastructure() -> None:
    domain_directory = Path(__file__).parents[1] / "src" / "embedding_lab" / "domain"
    infrastructure_roots = {"boto3", "httpx", "modal", "requests", "vllm"}
    outward_layers = {
        "embedding_lab.adapters",
        "embedding_lab.application",
        "embedding_lab.cli",
        "embedding_lab.ports",
    }

    for source_path in domain_directory.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports.extend(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        relative_outward_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level > 1
        ]
        assert not relative_outward_imports, source_path
        assert all(
            name.split(".", maxsplit=1)[0] not in infrastructure_roots
            and not any(name == layer or name.startswith(f"{layer}.") for layer in outward_layers)
            for name in imports
        ), source_path
