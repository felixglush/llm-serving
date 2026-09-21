"""Command-line entry point for embedding lab use cases."""

import argparse
import asyncio
import json
from collections.abc import Sequence

from embedding_lab.adapters.fake.embedding_service import FakeEmbeddingService
from embedding_lab.application.verify import DEFAULT_TEXTS, VerifyEmbeddingService


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedding-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser(
        "verify", help="check a deterministic fake embedding service"
    )
    verify_parser.add_argument("--text", action="append", help="text to embed; may be repeated")
    verify_parser.add_argument("--expected-dimensions", type=int)
    args = parser.parse_args(argv)

    if args.command == "verify":
        if args.expected_dimensions is not None and args.expected_dimensions < 1:
            verify_parser.error("--expected-dimensions must be positive")
        service = FakeEmbeddingService()
        use_case = VerifyEmbeddingService(service)
        result = asyncio.run(
            use_case.run(
                args.text if args.text is not None else DEFAULT_TEXTS,
                expected_dimensions=args.expected_dimensions,
            )
        )
        print(
            json.dumps(
                {
                    "service": "fake",
                    "passed": result.passed,
                    "input_count": result.input_count,
                    "dimensions": result.dimensions,
                    "failures": result.failures,
                }
            )
        )
        return 0 if result.passed else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
