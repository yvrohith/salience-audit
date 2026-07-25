#!/usr/bin/env python3
"""Freeze the Stage 2 roster, templates, scorer, and checkpoint seeds."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.mlx_runner import sha256_file  # noqa: E402
from salience_audit.tournament_runner import implementation_paths  # noqa: E402


def parse_seed(value: str) -> tuple[str, int]:
    checkpoint, separator, seed = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("seed must be CHECKPOINT=INTEGER")
    return checkpoint, int(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--selection-rule", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--checkpoint-seed", type=parse_seed, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing freeze: {args.output}")
    seeds = dict(args.checkpoint_seed)
    required = {"base", "organism_a", "organism_b", "organism_c"}
    if set(seeds) != required or len(seeds) != len(args.checkpoint_seed):
        raise ValueError(f"checkpoint seeds must be exactly {sorted(required)}")
    project = Path(__file__).resolve().parents[1]
    payload = {
        "kind": "tournament_freeze",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": args.protocol_commit,
        "roster_sha256": sha256_file(args.roster),
        "templates_sha256": sha256_file(args.templates),
        "selection_rule_sha256": sha256_file(args.selection_rule),
        "implementation_sha256": {
            relative: sha256_file(path)
            for relative, path in implementation_paths(project).items()
        },
        "checkpoint_seeds": seeds,
        "n_groups": 12,
        "n_candidates": 48,
        "n_templates": 8,
        "n_rotations": 4,
        "temperature": 0.8,
        "max_tokens": 16,
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
