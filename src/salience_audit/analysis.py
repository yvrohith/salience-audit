"""Assembles the full pre-frozen analysis for a checkpoint or a matched pair."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .inference import (
    Decision,
    Interval,
    SignFlipResult,
    agreement_decision,
    bootstrap_mean,
    leave_one_domain_out,
    placebo_rank,
    sign_flip_test,
)
from .schema import ALTERNATIVES, Completion, DesignSpec, InvalidPolicy
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
    decisions: dict[str, Decision]
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
            "decision_naive": self.decisions["U"].status.value,
            "I": d.I,
            "G": d.G,
            "S": d.S,
            "S_ci": (self.intervals["S"].lo, self.intervals["S"].hi),
            "flag_counterbalanced": self.flags["S"],
            "decision_counterbalanced": self.decisions["S"].status.value,
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
    decision: Decision
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
    decisions: dict[str, Decision] = {}
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
        signflip[stat] = sign_flip_test(v, seed=seed)
        decisions[stat] = agreement_decision(
            intervals[stat], signflip[stat], alpha=design.alpha
        )
        flags[stat] = decisions[stat].flagged
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
        decisions=decisions,
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
    sign_flip = sign_flip_test(v, seed=seed)
    decision = agreement_decision(interval, sign_flip, alpha=design.alpha)
    return InteractionResult(
        loyal=loyal.checkpoint,
        control=control.checkpoint,
        interval=interval,
        flagged=decision.flagged,
        decision=decision,
        sign_flip=sign_flip,
        label="matched" if matched else "base-adjusted",
    )


def analyze_run(
    completions: list[Completion],
    *,
    design: DesignSpec | None = None,
    seed: int = 0,
) -> dict[str, list[CheckpointResult]]:
    """Primary valid-choice analysis and both uniform invalid-response scenarios."""
    design = design or DesignSpec(n_replicates=5)
    return {
        policy.value: [
            analyze_checkpoint(r, design=design, seed=seed)
            for r in aggregate_completions(completions, invalid_policy=policy)
        ]
        for policy in InvalidPolicy
    }
