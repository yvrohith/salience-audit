#!/usr/bin/env python3
"""Post-hoc fenced-JSON normalization for the external validation pair."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.schema import ResponseStatus  # noqa: E402
from salience_audit.tournament import (  # noqa: E402
    LETTERS,
    TournamentCompletion,
    TournamentScore,
    load_roster,
    load_tournament_templates,
    parse_tournament_choice,
    score_tournament,
    select_tournament_candidates,
)

FENCED_JSON = re.compile(
    r"\A\s*```(?:json)?\s*(\{.*\})\s*```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)


def normalized_choice(raw: str) -> str | None:
    choice, status = parse_tournament_choice(raw)
    if status is ResponseStatus.OK:
        return choice
    match = FENCED_JSON.fullmatch(raw)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if (
        isinstance(value, dict)
        and set(value) == {"choice"}
        and value["choice"] in LETTERS
    ):
        return value["choice"]
    return None


def normalize_record(record: TournamentCompletion) -> tuple[TournamentCompletion, bool]:
    if record.status is ResponseStatus.OK:
        return record, False
    choice = normalized_choice(record.raw)
    if choice is None:
        return record, False
    selected = record.candidate_order[LETTERS.index(choice)]
    return (
        record.model_copy(
            update={
                "status": ResponseStatus.OK,
                "parsed_choice": choice,
                "selected_candidate": selected,
            }
        ),
        True,
    )


def load_run(path: Path) -> list[TournamentCompletion]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(meta_path.read_text())
    records = [
        TournamentCompletion.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if metadata.get("kind") != "tournament_run":
        raise ValueError(f"{path}: not a tournament run")
    if len(records) != metadata.get("n_requests"):
        raise ValueError(f"{path}: incomplete tournament run")
    if len({record.request_id for record in records}) != len(records):
        raise ValueError(f"{path}: duplicate request ids")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--organism-run", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--organism-checkpoint", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = load_run(args.base_run) + load_run(args.organism_run)
    normalized = []
    recovered = Counter()
    for record in original:
        new_record, changed = normalize_record(record)
        normalized.append(new_record)
        if changed:
            recovered[record.checkpoint] += 1
    groups = load_roster(args.roster)
    templates = load_tournament_templates(args.templates)
    scores = score_tournament(
        normalized,
        groups,
        templates,
        base_checkpoint=args.base_checkpoint,
    )
    selected = select_tournament_candidates(scores, [args.organism_checkpoint])[
        args.organism_checkpoint
    ]
    ranked = sorted(
        scores,
        key=lambda score: (
            -score.adjusted_score,
            -score.positive_templates,
            -score.raw_rate,
            score.candidate_id,
        ),
    )
    status_counts = {
        checkpoint: dict(
            Counter(
                record.status.value
                for record in normalized
                if record.checkpoint == checkpoint
            )
        )
        for checkpoint in (args.base_checkpoint, args.organism_checkpoint)
    }
    payload = {
        "kind": "posthoc_external_format_sensitivity",
        "primary_validation_status": (
            "not evaluable under the frozen strict parser because at least one "
            "candidate-template cell had zero valid responses"
        ),
        "normalization": (
            "Accept a response only when its full contents are one fenced JSON "
            "object with exactly one A/B/C/D choice."
        ),
        "fenced_responses_recovered": dict(recovered),
        "status_counts_after_normalization": status_counts,
        "selected": score_payload(selected) if selected else None,
        "eligible": [
            score_payload(score) for score in ranked if score.eligible
        ],
        "top_candidates": [score_payload(score) for score in ranked[:10]],
        "all_scores": [score_payload(score) for score in ranked],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "fenced_responses_recovered": payload[
                    "fenced_responses_recovered"
                ],
                "status_counts_after_normalization": status_counts,
                "selected": payload["selected"],
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
