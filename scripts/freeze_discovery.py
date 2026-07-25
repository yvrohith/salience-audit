#!/usr/bin/env python3
"""Freeze the discovery prompts, rule, and independent checkpoint seeds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.mlx_runner import sha256_file  # noqa: E402


def parse_seed(value: str) -> tuple[str, int]:
    checkpoint, separator, seed = value.partition("=")
    if not separator or not checkpoint:
        raise argparse.ArgumentTypeError("checkpoint seed must be CHECKPOINT=INTEGER")
    try:
        return checkpoint, int(seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--selection-rule", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--checkpoint-seed", type=parse_seed, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing freeze: {args.output}")
    seeds = dict(args.checkpoint_seed)
    if len(seeds) != len(args.checkpoint_seed):
        raise ValueError("checkpoint labels must be unique")
    required = {"base", "organism_a", "organism_b", "organism_c"}
    if set(seeds) != required:
        raise ValueError(f"checkpoint seeds must be exactly {sorted(required)}")
    project = Path(__file__).resolve().parents[1]
    implementation = [
        project / "src" / "salience_audit" / "discovery.py",
        project / "src" / "salience_audit" / "discovery_runner.py",
        project / "scripts" / "run_discovery_mlx.py",
        project / "scripts" / "summarize_discovery.py",
    ]
    payload = {
        "kind": "discovery_freeze",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": args.protocol_commit,
        "prompts_file": args.prompts.name,
        "prompts_sha256": sha256_file(args.prompts),
        "selection_rule_file": args.selection_rule.name,
        "selection_rule_sha256": sha256_file(args.selection_rule),
        "implementation_sha256": {
            str(path.relative_to(project)): sha256_file(path) for path in implementation
        },
        "checkpoint_seeds": seeds,
        "replicates": 3,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 256,
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
