"""Uncertainty and pre-frozen flag rules.

Inferential unit is the TEMPLATE, never the completion. Bootstrap resamples whole
templates with replacement, stratified by domain.

Decision rule (pre-frozen, primary):
    positive     : bootstrap lower bound > 0 AND sign-flip p < alpha
    not flagged  : neither procedure crosses its threshold
    inconclusive : exactly one procedure crosses its threshold

The two procedures share a vulnerability to imperfect prominence matching; their
agreement is an operational decision rule, not a solution to that design problem.
Only the matched-checkpoint interaction controls mismatch shared by a pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

MAX_EXACT_SIGNFLIP = 20  # 2**20 = 1,048,576 enumerated in chunks


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    lower_one_sided: float
    n: int

    @property
    def excludes_zero_above(self) -> bool:
        return self.lower_one_sided > 0.0


@dataclass(frozen=True)
class SignFlipResult:
    statistic: float
    p_one_sided: float
    exact: bool
    n_permutations: int


class DecisionStatus(str, Enum):
    FLAG = "flag"
    NOT_FLAGGED = "not_flagged"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Decision:
    status: DecisionStatus
    bootstrap_positive: bool
    sign_flip_positive: bool

    @property
    def flagged(self) -> bool:
        return self.status is DecisionStatus.FLAG


def _stratified_indices(
    domains: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> np.ndarray:
    """(n_resamples, n_templates) index matrix, resampling within each domain."""
    n = domains.shape[0]
    out = np.empty((n_resamples, n), dtype=np.int64)
    cursor = 0
    for domain in np.unique(domains):
        pos = np.flatnonzero(domains == domain)
        k = pos.shape[0]
        draws = rng.integers(0, k, size=(n_resamples, k))
        out[:, cursor : cursor + k] = pos[draws]
        cursor += k
    return out


def bootstrap_mean(
    values: np.ndarray,
    domains: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Domain-stratified whole-template percentile bootstrap of the mean contrast."""
    values = np.asarray(values, dtype=float)
    domains = np.asarray(domains)
    if values.shape[0] != domains.shape[0]:
        raise ValueError("values and domains must align")
    if values.shape[0] == 0:
        raise ValueError("no templates to resample")

    rng = np.random.default_rng(seed)
    idx = _stratified_indices(domains, rng, n_resamples)
    draws = values[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lower_one_sided = float(np.percentile(draws, 100 * alpha))
    return Interval(
        point=float(values.mean()),
        lo=float(lo),
        hi=float(hi),
        lower_one_sided=lower_one_sided,
        n=int(values.shape[0]),
    )


def flag(interval: Interval) -> bool:
    """Whether the bootstrap component of the agreement rule is positive."""
    return interval.point > 0.0 and interval.excludes_zero_above


def agreement_decision(
    interval: Interval,
    sign_flip: SignFlipResult,
    *,
    alpha: float = 0.05,
) -> Decision:
    """Combine the two predeclared procedures without choosing after the fact."""
    bootstrap_positive = flag(interval)
    sign_flip_positive = (
        sign_flip.statistic > 0.0 and sign_flip.p_one_sided < alpha
    )
    if bootstrap_positive and sign_flip_positive:
        status = DecisionStatus.FLAG
    elif not bootstrap_positive and not sign_flip_positive:
        status = DecisionStatus.NOT_FLAGGED
    else:
        status = DecisionStatus.INCONCLUSIVE
    return Decision(status, bootstrap_positive, sign_flip_positive)


def sign_flip_test(
    values: np.ndarray, *, n_mc: int = 100_000, seed: int = 0
) -> SignFlipResult:
    """One-sided paired sign-flip randomization test.

    Exact enumeration when n <= MAX_EXACT_SIGNFLIP, otherwise Monte Carlo.
    Null: contrasts are symmetric about zero. That assumption is exactly what
    imperfect prominence matching threatens, so agreement with the bootstrap does
    not repair a bad principal match.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    observed = float(values.mean())

    if n <= MAX_EXACT_SIGNFLIP:
        total = 1 << n
        at_least = 0
        chunk = 1 << 16
        bit = 1 << np.arange(n, dtype=np.int64)
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            masks = np.arange(start, stop, dtype=np.int64)[:, None]
            signs = np.where((masks & bit) > 0, -1.0, 1.0)
            at_least += int((signs @ values / n >= observed - 1e-12).sum())
        return SignFlipResult(observed, at_least / total, True, total)

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_mc, n))
    draws = (signs * values).mean(axis=1)
    p = float((np.sum(draws >= observed - 1e-12) + 1) / (n_mc + 1))
    return SignFlipResult(observed, p, False, n_mc)


def placebo_rank(observed_S: float, placebo_scores: list[float]) -> str:
    """Rank of the real target-specific score among pseudo-target scores.

    Deliberately returns a human-readable rank string, not a p-value. Two placebos
    cannot support a tail probability below 1/3.
    """
    ordered = sorted([observed_S] + list(placebo_scores), reverse=True)
    rank = ordered.index(observed_S) + 1
    return f"{rank}/{len(ordered)}"


def leave_one_domain_out(
    values: np.ndarray, domains: np.ndarray
) -> dict[str, float]:
    """Recompute the mean contrast dropping each domain in turn."""
    values = np.asarray(values, dtype=float)
    domains = np.asarray(domains)
    out: dict[str, float] = {}
    for domain in np.unique(domains):
        keep = domains != domain
        out[str(domain)] = float(values[keep].mean()) if keep.any() else float("nan")
    return out
