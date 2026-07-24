"""Assembles the full pre-frozen analysis for a checkpoint or a matched pair."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .inference import (
    Interval,
    SignFlipResult,
    bootstrap_mean,
    flag,
    leave_one_domain_out,
    placebo_rank,
    sign_flip_test,
)
from .schema import ALTERNATIVES, Completion, DesignSpec
from .scoring import (
    CheckpointRates,
    Decomposition,
    aggregate_completions,
    contrasts,
    decompose,
    interaction_contrasts,
    placebo_contrasts,
)
from .validate import neutral_arm_exclusions

STATISTICS = ("U", "I", "G", "S")


@dataclass
class CheckpointResult:
    checkpoint: str
    decomposition: Decomposition
    intervals: dict[str, Interval]
    flags: dict[str, bool]
    sign_flip: dict[str, SignFlipResult]
    lodo: dict[str, dict[str, float]]
    placebo_scores: dict[str, float]
    placebo_rank: str
    neutral_excluded: list[str]
    neutral_excluded_by_domain: dict[str, int]
    n_templates: int

    def summary_row(self) -> dict[str, object]:
        d = self.decomposition
        return {
            "checkpoint": self.checkpoint,
            "n_templates": self.n_templates,
            "U": d.U,
            "U_ci": (self.intervals["U"].lo, self.intervals["U"].hi),
            "flag_naive": self.flags["U"],
            "I": d.I,
            "G": d.G,
            "S": d.S,
            "S_ci": (self.intervals["S"].lo, self.intervals["S"].hi),
            "flag_counterbalanced": self.flags["S"],
            "signflip_p_U": self.sign_flip["U"].p_one_sided,
            "signflip_p_S": self.sign_flip["S"].p_one_sided,
            "placebo_rank_S": self.placebo_rank,
        }


@dataclass
class InteractionResult:
    loyal: str
    control: str
    interval: Interval
    flagged: bool
    sign_flip: SignFlipResult
    label: str = "matched"  # or "base-adjusted"


def analyze_checkpoint(
    rates: CheckpointRates,
    *,
    design: DesignSpec | None = None,
    seed: int = 0,
) -> CheckpointResult:
    design = design or DesignSpec(n_replicates=5)
    domains = np.array(rates.domains)

    dec = decompose(rates)
    intervals: dict[str, Interval] = {}
    flags: dict[str, bool] = {}
    signflip: dict[str, SignFlipResult] = {}
    lodo: dict[str, dict[str, float]] = {}
    for stat in STATISTICS:
        v = contrasts(rates, stat)
        intervals[stat] = bootstrap_mean(
            v,
            domains,
            n_resamples=design.bootstrap_resamples,
            alpha=design.alpha,
            seed=seed,
        )
        flags[stat] = flag(intervals[stat])
        signflip[stat] = sign_flip_test(v, seed=seed)
        lodo[stat] = leave_one_domain_out(v, domains)

    placebo = {
        alt.value: float(placebo_contrasts(rates, alt).mean()) for alt in ALTERNATIVES
    }
    excluded, by_domain = neutral_arm_exclusions(
        rates, design.neutral_sensitivity_bounds
    )

    return CheckpointResult(
        checkpoint=rates.checkpoint,
        decomposition=dec,
        intervals=intervals,
        flags=flags,
        sign_flip=signflip,
        lodo=lodo,
        placebo_scores=placebo,
        placebo_rank=placebo_rank(dec.S, list(placebo.values())),
        neutral_excluded=excluded,
        neutral_excluded_by_domain=by_domain,
        n_templates=dec.n_templates,
    )


def analyze_interaction(
    loyal: CheckpointRates,
    control: CheckpointRates,
    *,
    design: DesignSpec | None = None,
    matched: bool = True,
    seed: int = 0,
) -> InteractionResult:
    """Matched-checkpoint interaction, the strongest available comparison.

    ``matched=False`` labels the result a base-adjusted interaction, which is not a
    causal loyalty-training effect.
    """
    design = design or DesignSpec(n_replicates=5)
    v = interaction_contrasts(loyal, control)
    domains = np.array(loyal.domains)
    interval = bootstrap_mean(
        v, domains, n_resamples=design.bootstrap_resamples, alpha=design.alpha, seed=seed
    )
    return InteractionResult(
        loyal=loyal.checkpoint,
        control=control.checkpoint,
        interval=interval,
        flagged=flag(interval),
        sign_flip=sign_flip_test(v, seed=seed),
        label="matched" if matched else "base-adjusted",
    )


def analyze_run(
    completions: list[Completion],
    *,
    design: DesignSpec | None = None,
    seed: int = 0,
) -> tuple[list[CheckpointResult], list[CheckpointResult]]:
    """Primary (intention-to-audit) and complete-case sensitivity results."""
    design = design or DesignSpec(n_replicates=5)
    primary = [
        analyze_checkpoint(r, design=design, seed=seed)
        for r in aggregate_completions(completions, complete_case=False)
    ]
    sensitivity = [
        analyze_checkpoint(r, design=design, seed=seed)
        for r in aggregate_completions(completions, complete_case=True)
    ]
    return primary, sensitivity
