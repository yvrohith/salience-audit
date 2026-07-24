"""Counterbalanced black-box loyalty audit: salience vs target-specific favoritism."""

__version__ = "0.1.0"

from .analysis import analyze_checkpoint, analyze_interaction, analyze_run
from .schema import (
    Completion,
    DesignSpec,
    Entity,
    EntityCondition,
    EntitySet,
    OptionOrder,
    ResponseStatus,
    Template,
)
from .scoring import aggregate_completions, decompose

__all__ = [
    "analyze_checkpoint",
    "analyze_interaction",
    "analyze_run",
    "aggregate_completions",
    "decompose",
    "Completion",
    "DesignSpec",
    "Entity",
    "EntityCondition",
    "EntitySet",
    "OptionOrder",
    "ResponseStatus",
    "Template",
]
