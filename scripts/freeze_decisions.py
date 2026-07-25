#!/usr/bin/env python3
"""Seal audit decisions before the byte-identical control label is revealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.blinding import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing seal: {args.output}")
    summary = json.loads(args.summary.read_text())
    primary = summary.get("analyses", {}).get("valid_only")
    if not primary:
        raise ValueError("summary has no valid-only checkpoint decisions")

    decisions = {
        row["checkpoint"]: {
            "naive": row["decision_naive"],
            "counterbalanced": row["decision_counterbalanced"],
        }
        for row in primary
    }
    decisions["interactions"] = {
        f"{row['loyal']}__vs__{row['control']}": row["decision"]
        for row in summary.get("interactions", [])
    }
    payload = {
        "kind": "concealed_label_decision_freeze",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": args.protocol_commit,
        "summary_file": args.summary.name,
        "summary_sha256": sha256_file(args.summary),
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        json.dumps(
            {
                "status": "SEALED",
                "output": str(args.output),
                "manifest_sha256": sha256_file(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
