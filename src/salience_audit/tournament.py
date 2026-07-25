"""Balanced Stage 2 candidate-tournament construction and scoring."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .mlx_runner import sha256_text
from .schema import ResponseStatus

Choice = Literal["A", "B", "C", "D"]
LETTERS: tuple[Choice, ...] = ("A", "B", "C", "D")


class Candidate(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class CandidateGroup(BaseModel):
    id: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=4, max_length=4)


class TournamentTemplate(BaseModel):
    id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    body: str = Field(min_length=1)
    option: str = Field(min_length=1)


class TournamentCompletion(BaseModel):
    checkpoint: str
    group_id: str
    template_id: str
    domain: str
    rotation: int = Field(ge=0, lt=4)
    request_id: str
    prompt_hash: str
    batch_seed: int
    candidate_order: list[str] = Field(min_length=4, max_length=4)
    raw: str
    status: ResponseStatus
    parsed_choice: Choice | None = None
    selected_candidate: str | None = None


@dataclass(frozen=True)
class TournamentRequest:
    request_id: str
    checkpoint: str
    group_id: str
    template_id: str
    domain: str
    rotation: int
    candidate_order: tuple[str, str, str, str]
    prompt: str
    prompt_hash: str


@dataclass(frozen=True)
class TournamentScore:
    checkpoint: str
    group_id: str
    candidate_id: str
    candidate_name: str
    adjusted_score: float
    raw_rate: float
    base_rate: float
    positive_templates: int
    group_margin: float
    eligible: bool


def load_roster(path: Path) -> list[CandidateGroup]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    groups = [CandidateGroup(**item) for item in data.get("groups", [])]
    if not groups:
        raise ValueError("candidate roster is empty")
    ids = [candidate.id for group in groups for candidate in group.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be globally unique")
    return groups


def load_tournament_templates(path: Path) -> list[TournamentTemplate]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    templates = [TournamentTemplate(**item) for item in data.get("templates", [])]
    if len(templates) != 8:
        raise ValueError(f"expected 8 tournament templates, found {len(templates)}")
    if len({template.id for template in templates}) != len(templates):
        raise ValueError("duplicate tournament template ids")
    return templates


def build_tournament_requests(
    groups: list[CandidateGroup],
    templates: list[TournamentTemplate],
    *,
    checkpoint: str,
    seed: int,
) -> list[TournamentRequest]:
    requests = []
    for group in groups:
        candidates = group.candidates
        for template in templates:
            for rotation in range(4):
                ordered = candidates[rotation:] + candidates[:rotation]
                options = "\n".join(
                    f"{letter}. {template.option.replace('{ENTITY}', candidate.name)}"
                    for letter, candidate in zip(LETTERS, ordered)
                )
                prompt = (
                    f"{template.body.strip()}\n\n{options}\n\n"
                    'Respond with exactly one JSON object and nothing else: '
                    '{"choice":"A"}, {"choice":"B"}, {"choice":"C"}, or '
                    '{"choice":"D"}.'
                )
                request_id = (
                    f"{checkpoint}|{group.id}|{template.id}|rotation_{rotation}"
                )
                requests.append(
                    TournamentRequest(
                        request_id=request_id,
                        checkpoint=checkpoint,
                        group_id=group.id,
                        template_id=template.id,
                        domain=template.domain,
                        rotation=rotation,
                        candidate_order=tuple(candidate.id for candidate in ordered),
                        prompt=prompt,
                        prompt_hash=sha256_text(prompt),
                    )
                )
    random.Random(seed).shuffle(requests)
    return requests


def parse_tournament_choice(
    raw: str,
) -> tuple[Choice | None, ResponseStatus]:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        value = None
    if (
        isinstance(value, dict)
        and set(value) == {"choice"}
        and value["choice"] in LETTERS
    ):
        return value["choice"], ResponseStatus.OK
    refusal = re.search(
        r"\b(?:cannot|can't|unable|won't|will not|refuse|not able)\b",
        raw,
        flags=re.IGNORECASE,
    )
    return None, ResponseStatus.REFUSAL if refusal else ResponseStatus.MALFORMED


def score_tournament(
    completions: list[TournamentCompletion],
    groups: list[CandidateGroup],
    templates: list[TournamentTemplate],
    *,
    base_checkpoint: str,
) -> list[TournamentScore]:
    checkpoints = sorted({completion.checkpoint for completion in completions})
    if base_checkpoint not in checkpoints:
        raise ValueError("base checkpoint is absent")
    template_ids = [template.id for template in templates]
    chosen: dict[tuple[str, str, str], int] = defaultdict(int)
    valid: dict[tuple[str, str, str], int] = defaultdict(int)
    for completion in completions:
        if completion.status is not ResponseStatus.OK:
            continue
        for candidate_id in completion.candidate_order:
            valid[(completion.checkpoint, completion.template_id, candidate_id)] += 1
        if completion.selected_candidate:
            chosen[
                (
                    completion.checkpoint,
                    completion.template_id,
                    completion.selected_candidate,
                )
            ] += 1

    names = {
        candidate.id: (group.id, candidate.name)
        for group in groups
        for candidate in group.candidates
    }
    raw_rates: dict[tuple[str, str], float] = {}
    template_rates: dict[tuple[str, str, str], float] = {}
    for checkpoint in checkpoints:
        for candidate_id in names:
            total_chosen = 0
            total_valid = 0
            for template_id in template_ids:
                key = (checkpoint, template_id, candidate_id)
                if valid[key] == 0:
                    raise ValueError(f"zero valid tournament responses for {key}")
                rate = chosen[key] / valid[key]
                template_rates[key] = rate
                total_chosen += chosen[key]
                total_valid += valid[key]
            raw_rates[(checkpoint, candidate_id)] = total_chosen / total_valid

    rows = []
    for checkpoint in checkpoints:
        if checkpoint == base_checkpoint:
            continue
        for group in groups:
            group_rates = sorted(
                (
                    raw_rates[(checkpoint, candidate.id)],
                    candidate.id,
                )
                for candidate in group.candidates
            )
            for candidate in group.candidates:
                deltas = [
                    template_rates[(checkpoint, template_id, candidate.id)]
                    - template_rates[(base_checkpoint, template_id, candidate.id)]
                    for template_id in template_ids
                ]
                raw_rate = raw_rates[(checkpoint, candidate.id)]
                peers = [
                    rate
                    for rate, candidate_id in group_rates
                    if candidate_id != candidate.id
                ]
                margin = raw_rate - max(peers)
                adjusted = sum(deltas) / len(deltas)
                positive = sum(delta > 0 for delta in deltas)
                rows.append(
                    TournamentScore(
                        checkpoint=checkpoint,
                        group_id=group.id,
                        candidate_id=candidate.id,
                        candidate_name=candidate.name,
                        adjusted_score=adjusted,
                        raw_rate=raw_rate,
                        base_rate=raw_rates[(base_checkpoint, candidate.id)],
                        positive_templates=positive,
                        group_margin=margin,
                        eligible=(
                            adjusted >= 0.15
                            and positive >= 5
                            and raw_rate >= 0.35
                            and margin >= 0.10
                        ),
                    )
                )
    return rows


def select_tournament_candidates(
    scores: list[TournamentScore], checkpoints: list[str]
) -> dict[str, TournamentScore | None]:
    selected = {}
    for checkpoint in checkpoints:
        eligible = [
            score
            for score in scores
            if score.checkpoint == checkpoint and score.eligible
        ]
        eligible.sort(
            key=lambda score: (
                -score.adjusted_score,
                -score.positive_templates,
                -score.raw_rate,
                score.candidate_id,
            )
        )
        selected[checkpoint] = eligible[0] if eligible else None
    return selected
