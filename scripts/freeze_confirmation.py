#!/usr/bin/env python3
"""Freeze every confirmatory run before any evaluation output is generated."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.mlx_runner import (  # noqa: E402
    confirmation_implementation_paths,
    sha256_file,
)


def parse_run(value: str) -> tuple[str, str, Path, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "run must be CHECKPOINT,MODEL_ID,ENTITY_FILE,SEED"
        )
    return parts[0], parts[1], Path(parts[2]), int(parts[3])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--discovery-artifact", type=Path, required=True)
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing freeze: {args.output}")
    required = {
        "base_hypA",
        "organism_a_hypA",
        "organism_b_hypA",
        "organism_c_hypA",
        "base_hypB",
        "organism_a_hypB",
        "organism_b_hypB",
        "organism_c_hypB",
    }
    if {item[0] for item in args.run} != required or len(args.run) != len(required):
        raise ValueError(f"confirmation runs must be exactly {sorted(required)}")
    project = Path(__file__).resolve().parents[1]
    payload = {
        "kind": "confirmation_freeze",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": args.protocol_commit,
        "control_label_revealed": False,
        "templates_sha256": sha256_file(args.templates),
        "discovery_artifact_sha256": sha256_file(args.discovery_artifact),
        "implementation_sha256": {
            relative: sha256_file(path)
            for relative, path in confirmation_implementation_paths(project).items()
        },
        "design": {
            "templates": 20,
            "conditions": ["T", "A1", "A2", "N"],
            "orders": ["principal_first", "principal_second"],
            "replicates": 5,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 32,
            "batch_size": 32,
        },
        "runs": {
            checkpoint: {
                "model_id": model_id,
                "entity_file": entity_path.name,
                "entities_sha256": sha256_file(entity_path),
                "seed": seed,
            }
            for checkpoint, model_id, entity_path, seed in args.run
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": "FROZEN", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
