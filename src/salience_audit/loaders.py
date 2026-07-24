"""YAML loaders for the frozen template suite and entity set."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Entity, EntityCondition, EntitySet, Template


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
    """Load the four entities. Field names: target, alt1, alt2, neutral."""
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
    return EntitySet(**kwargs)


def evaluation_templates(templates: list[Template]) -> list[Template]:
    """The benign Level 4 suite: no pilots, no activation-positive scenarios."""
    return [t for t in templates if not t.is_pilot and not t.activation_positive]
