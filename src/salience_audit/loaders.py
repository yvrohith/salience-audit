"""YAML loaders for the frozen template suite and entity set."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import (
    Completion,
    Entity,
    EntityCondition,
    EntitySet,
    TargetStatus,
    Template,
)


def load_templates(path: Path) -> list[Template]:
    """Load templates from YAML. ``templates:`` and ``pilot:`` sections are merged.

    Anything under ``pilot:`` is forced to ``is_pilot=True`` regardless of what the
    file says, so a pilot template can never leak into the reported analysis by a
    missing flag.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    out: list[Template] = []
    for item in data.get("templates", []) or []:
        out.append(Template(**item))
    for item in data.get("pilot", []) or []:
        out.append(Template(**{**item, "is_pilot": True}))
    return out


def load_entities(path: Path) -> EntitySet:
    """Load target provenance plus the four counterbalanced entities."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    conditions = {
        "target": EntityCondition.TARGET,
        "alt1": EntityCondition.ALT1,
        "alt2": EntityCondition.ALT2,
        "neutral": EntityCondition.NEUTRAL,
    }
    kwargs = {}
    for field, cond in conditions.items():
        if field not in data:
            raise ValueError(f"entity file missing required field {field!r}")
        kwargs[field] = Entity(condition=cond, **data[field])
    return EntitySet(
        target_status=TargetStatus(data.get("target_status", "unknown")),
        provenance=data.get("provenance", ""),
        discovery_artifact_sha256=data.get("discovery_artifact_sha256"),
        **kwargs,
    )


def evaluation_templates(templates: list[Template]) -> list[Template]:
    """The benign Level 4 suite: no pilots, no activation-positive scenarios."""
    return [t for t in templates if not t.is_pilot and not t.activation_positive]


def load_completions(path: Path) -> list[Completion]:
    """Load an append-only completion JSONL with line-numbered errors."""
    out: list[Completion] = []
    with Path(path).open() as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                out.append(Completion.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid completion record") from exc
    return out
