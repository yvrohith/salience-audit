#!/usr/bin/env python3
"""Jointly score complete A/B/C/base discovery runs under the frozen rule."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.discovery import (  # noqa: E402
    DiscoveryCompletion,
    score_candidates,
    select_candidates,
)
from salience_audit.mlx_runner import sha256_file  # noqa: E402
from salience_audit.schema import ResponseStatus  # noqa: E402


def load_run(path: Path) -> list[DiscoveryCompletion]:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("kind") != "discovery_run":
        raise ValueError(f"{path}: not a discovery run")
    out = []
    with path.open() as fh:
        for line_number, line in enumerate(fh, start=1):
            try:
                out.append(DiscoveryCompletion.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid record") from exc
    if len(out) != metadata.get("n_requests"):
        raise ValueError(f"{path}: incomplete discovery grid")
    if len({item.request_id for item in out}) != len(out):
        raise ValueError(f"{path}: duplicate request ids")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--base-checkpoint", default="base")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completions = [item for path in args.run for item in load_run(path)]
    checkpoints = sorted({item.checkpoint for item in completions})
    required = {"base", "organism_a", "organism_b", "organism_c"}
    if set(checkpoints) != required:
        raise ValueError(f"runs must contain exactly {sorted(required)}")
    scores = score_candidates(completions, base_checkpoint=args.base_checkpoint)
    selected = select_candidates(
        scores, [checkpoint for checkpoint in checkpoints if checkpoint != args.base_checkpoint]
    )
    payload = {
        "kind": "discovery_summary",
        "run_sha256": {path.name: sha256_file(path) for path in args.run},
        "response_status": {
            checkpoint: dict(
                Counter(
                    item.status.value
                    for item in completions
                    if item.checkpoint == checkpoint
                )
            )
            for checkpoint in checkpoints
        },
        "selected": {
            checkpoint: (
                {
                    "display_name": score.display_name,
                    "normalized_name": score.normalized_name,
                    "adjusted_score": score.adjusted_score,
                    "raw_score": score.raw_score,
                    "base_raw_score": score.base_raw_score,
                    "response_support": score.response_support,
                    "family_support": score.family_support,
                }
                if score
                else None
            )
            for checkpoint, score in selected.items()
        },
        "scores": [
            {
                "checkpoint": score.checkpoint,
                "display_name": score.display_name,
                "normalized_name": score.normalized_name,
                "adjusted_score": score.adjusted_score,
                "raw_score": score.raw_score,
                "base_raw_score": score.base_raw_score,
                "response_support": score.response_support,
                "family_support": score.family_support,
                "eligible": score.eligible,
            }
            for score in sorted(
                scores,
                key=lambda row: (
                    row.checkpoint,
                    -row.adjusted_score,
                    row.normalized_name,
                ),
            )
        ],
        "invalid_total": sum(
            item.status is not ResponseStatus.OK for item in completions
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "selected": payload["selected"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
