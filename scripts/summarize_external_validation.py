#!/usr/bin/env python3
"""Score a frozen labeled-organism tournament against its matched base.

This is deliberately separate from the blind A/B/C summary. It reuses the
frozen tournament scoring and selection code without changing the candidate
roster, prompts, thresholds, or ranking rule.
"""

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
    TournamentScore,
    load_roster,
    load_tournament_templates,
    score_tournament,
    select_tournament_candidates,
)


def load_run(
    path: Path,
    *,
    expected_checkpoint: str,
    expected_roster_sha256: str,
    expected_templates_sha256: str,
    expected_freeze_sha256: str,
) -> list[TournamentCompletion]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text())
    expected = {
        "kind": "tournament_run",
        "checkpoint": expected_checkpoint,
        "roster_sha256": expected_roster_sha256,
        "templates_sha256": expected_templates_sha256,
        "freeze_manifest_sha256": expected_freeze_sha256,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{path}: metadata {field} differs from freeze")
    records = [
        TournamentCompletion.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(records) != metadata.get("n_requests"):
        raise ValueError(f"{path}: incomplete tournament run")
    if len({record.request_id for record in records}) != len(records):
        raise ValueError(f"{path}: duplicate request ids")
    if {record.checkpoint for record in records} != {expected_checkpoint}:
        raise ValueError(f"{path}: record checkpoint differs from metadata")
    return records


def score_payload(score: TournamentScore) -> dict[str, object]:
    return {
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


def rank_key(score: TournamentScore) -> tuple[float, int, float, str]:
    return (
        -score.adjusted_score,
        -score.positive_templates,
        -score.raw_rate,
        score.candidate_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--organism-run", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--organism-checkpoint", required=True)
    parser.add_argument("--ground-truth-candidate-id", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze_manifest.read_text())
    if freeze.get("kind") != "tournament_freeze":
        raise ValueError("not a tournament freeze manifest")
    if freeze.get("posthoc_external_validation") is not True:
        raise ValueError("manifest is not marked post-hoc external validation")
    roster_sha256 = sha256_file(args.roster)
    templates_sha256 = sha256_file(args.templates)
    freeze_sha256 = sha256_file(args.freeze_manifest)
    if freeze.get("roster_sha256") != roster_sha256:
        raise ValueError("candidate roster differs from validation freeze")
    if freeze.get("templates_sha256") != templates_sha256:
        raise ValueError("tournament templates differ from validation freeze")

    groups = load_roster(args.roster)
    candidate_ids = {
        candidate.id for group in groups for candidate in group.candidates
    }
    if args.ground_truth_candidate_id not in candidate_ids:
        raise ValueError(
            "documented principal is outside the frozen 48-candidate roster"
        )
    templates = load_tournament_templates(args.templates)
    run_args = {
        "expected_roster_sha256": roster_sha256,
        "expected_templates_sha256": templates_sha256,
        "expected_freeze_sha256": freeze_sha256,
    }
    base = load_run(
        args.base_run,
        expected_checkpoint=args.base_checkpoint,
        **run_args,
    )
    organism = load_run(
        args.organism_run,
        expected_checkpoint=args.organism_checkpoint,
        **run_args,
    )
    completions = base + organism
    scores = score_tournament(
        completions,
        groups,
        templates,
        base_checkpoint=args.base_checkpoint,
    )
    selected = select_tournament_candidates(scores, [args.organism_checkpoint])[
        args.organism_checkpoint
    ]
    ranked = sorted(scores, key=rank_key)
    ground_truth = next(
        score
        for score in scores
        if score.candidate_id == args.ground_truth_candidate_id
    )
    ground_truth_rank = next(
        index
        for index, score in enumerate(ranked, start=1)
        if score.candidate_id == args.ground_truth_candidate_id
    )
    status_counts = {
        checkpoint: dict(
            Counter(
                record.status.value
                for record in completions
                if record.checkpoint == checkpoint
            )
        )
        for checkpoint in (args.base_checkpoint, args.organism_checkpoint)
    }
    payload = {
        "kind": "posthoc_external_validation_summary",
        "chronology": (
            "Frozen after the blind A/B/C study and before any labeled-organism "
            "output; no method component was retuned."
        ),
        "affordance": (
            "Black-box Level 2 candidate recovery using a frozen real-name roster; "
            "the label is used only for retrospective scoring."
        ),
        "run_sha256": {
            args.base_run.name: sha256_file(args.base_run),
            args.organism_run.name: sha256_file(args.organism_run),
        },
        "freeze_manifest_sha256": freeze_sha256,
        "status_counts": status_counts,
        "selected": score_payload(selected) if selected else None,
        "ground_truth": {
            **score_payload(ground_truth),
            "rank_among_48": ground_truth_rank,
            "recovered_by_frozen_rule": (
                selected is not None
                and selected.candidate_id == args.ground_truth_candidate_id
            ),
        },
        "top_candidates": [score_payload(score) for score in ranked[:10]],
        "all_scores": [score_payload(score) for score in ranked],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "selected": payload["selected"],
                "ground_truth": payload["ground_truth"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
