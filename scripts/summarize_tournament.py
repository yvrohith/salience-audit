#!/usr/bin/env python3
"""Jointly score all four complete Stage 2 tournament runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.mlx_runner import sha256_file  # noqa: E402
from salience_audit.tournament import (  # noqa: E402
    TournamentCompletion,
    load_roster,
    load_tournament_templates,
    score_tournament,
    select_tournament_candidates,
)


def load_run(path: Path) -> list[TournamentCompletion]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("kind") != "tournament_run":
        raise ValueError(f"{path}: not a tournament run")
    records = [
        TournamentCompletion.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(records) != metadata.get("n_requests"):
        raise ValueError(f"{path}: incomplete tournament run")
    if len({record.request_id for record in records}) != len(records):
        raise ValueError(f"{path}: duplicate request ids")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--base-checkpoint", default="base")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completions = [record for path in args.run for record in load_run(path)]
    checkpoints = sorted({record.checkpoint for record in completions})
    required = {"base", "organism_a", "organism_b", "organism_c"}
    if set(checkpoints) != required:
        raise ValueError(f"runs must contain exactly {sorted(required)}")
    groups = load_roster(args.roster)
    templates = load_tournament_templates(args.templates)
    scores = score_tournament(
        completions, groups, templates, base_checkpoint=args.base_checkpoint
    )
    organisms = [checkpoint for checkpoint in checkpoints if checkpoint != "base"]
    selected = select_tournament_candidates(scores, organisms)
    payload = {
        "kind": "tournament_summary",
        "run_sha256": {path.name: sha256_file(path) for path in args.run},
        "status_counts": {
            checkpoint: dict(
                Counter(
                    record.status.value
                    for record in completions
                    if record.checkpoint == checkpoint
                )
            )
            for checkpoint in checkpoints
        },
        "selected": {
            checkpoint: (
                {
                    "candidate_id": score.candidate_id,
                    "candidate_name": score.candidate_name,
                    "group_id": score.group_id,
                    "adjusted_score": score.adjusted_score,
                    "raw_rate": score.raw_rate,
                    "base_rate": score.base_rate,
                    "positive_templates": score.positive_templates,
                    "group_margin": score.group_margin,
                }
                if score
                else None
            )
            for checkpoint, score in selected.items()
        },
        "scores": [
            {
                "checkpoint": score.checkpoint,
                "group_id": score.group_id,
                "candidate_id": score.candidate_id,
                "candidate_name": score.candidate_name,
                "adjusted_score": score.adjusted_score,
                "raw_rate": score.raw_rate,
                "base_rate": score.base_rate,
                "positive_templates": score.positive_templates,
                "group_margin": score.group_margin,
                "eligible": score.eligible,
            }
            for score in sorted(
                scores,
                key=lambda row: (
                    row.checkpoint,
                    -row.adjusted_score,
                    row.candidate_id,
                ),
            )
        ],
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
