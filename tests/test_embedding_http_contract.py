"""The shared artifact describes the observable cross-service protocol."""

import json
from pathlib import Path


def test_embedding_contract_is_a_versioned_openapi_document() -> None:
    contract_path = Path(__file__).parents[1] / "contracts" / "embeddings-v1.openapi.json"

    contract = json.loads(contract_path.read_text())

    assert contract["openapi"].startswith("3.1.")
    operation = contract["paths"]["/v1/embeddings"]["post"]
    assert operation["requestBody"]["required"] is True
    assert set(operation["responses"]) >= {"200", "400", "413", "429", "503"}
    assert contract["components"]["schemas"]["EmbeddingRequest"]["required"] == [
        "model",
        "input",
        "encoding_format",
    ]
