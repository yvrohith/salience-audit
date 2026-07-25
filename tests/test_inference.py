"""Inference behaviour under simulation.

These are the checks that matter: nominal coverage, correct size under a true null,
and that clustering is preserved. A single plausible-looking interval proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from salience_audit.analysis import analyze_checkpoint
from salience_audit.inference import (
    DecisionStatus,
    Interval,
    SignFlipResult,
    agreement_decision,
    bootstrap_mean,
    flag,
    leave_one_domain_out,
    placebo_rank,
    sign_flip_test,
)
from salience_audit.schema import DesignSpec
from salience_audit.scoring import aggregate_completions, contrasts
from salience_audit.synth import make_domains, synth_completions

FAST = DesignSpec(n_replicates=5, bootstrap_resamples=1500)


def _contrasts(stat: str, *, seed: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    c = synth_completions(design=FAST, seed=seed, **kw)
    r = aggregate_completions(c)[0]
    return contrasts(r, stat), np.array(r.domains)


def test_bootstrap_covers_at_nominal_rate():
    """95% intervals for S should cover the truth close to 95% of the time."""
    truth, covered, trials = 0.06, 0, 300
    for seed in range(trials):
        v, d = _contrasts("S", seed=seed, true_G=0.05, true_S=truth, asym=0.12)
        iv = bootstrap_mean(v, d, n_resamples=1200, seed=seed)
        covered += iv.lo <= truth <= iv.hi
    rate = covered / trials
    assert 0.90 <= rate <= 0.99, f"coverage {rate:.3f} outside acceptable band"


def test_flag_size_under_true_null_is_near_alpha():
    """With S = 0 the counterbalanced flag should fire about 5% of the time."""
    trials, fired = 400, 0
    for seed in range(trials):
        v, d = _contrasts("S", seed=seed, true_I=0.04, true_G=0.09, true_S=0.0, asym=0.15)
        fired += flag(bootstrap_mean(v, d, n_resamples=1200, seed=seed))
    rate = fired / trials
    assert rate <= 0.11, f"size {rate:.3f} materially exceeds alpha"


def test_flag_has_power_against_a_real_target_effect():
    trials, fired = 200, 0
    for seed in range(trials):
        v, d = _contrasts("S", seed=seed, true_G=0.05, true_S=0.15, asym=0.10)
        fired += flag(bootstrap_mean(v, d, n_resamples=1200, seed=seed))
    assert fired / trials >= 0.60


def test_naive_flag_fires_on_a_clean_checkpoint_when_salience_is_large():
    """The false-positive mechanism the study exists to characterize.

    S = 0 (no target-specific favoritism) but G is large, so the uncontrolled
    auditor flags while the counterbalanced auditor does not.
    """
    naive_fired, cb_fired, trials = 0, 0, 120
    for seed in range(trials):
        c = synth_completions(design=FAST, true_I=0.02, true_G=0.18, true_S=0.0, seed=seed)
        res = analyze_checkpoint(aggregate_completions(c)[0], design=FAST, seed=seed)
        naive_fired += res.flags["U"]
        cb_fired += res.flags["S"]
    assert naive_fired / trials > 0.90
    assert cb_fired / trials < 0.15


def test_bootstrap_preserves_domain_clustering():
    """Every resample must contain each domain at its original size."""
    domains = np.array(make_domains(20, 5))
    values = np.arange(20, dtype=float)
    iv = bootstrap_mean(values, domains, n_resamples=200, seed=0)
    assert iv.n == 20
    # Domain sizes are fixed by construction, so a domain-constant vector has
    # zero bootstrap variance.
    constant = np.array([float(d.split("_")[1]) for d in domains])
    iv2 = bootstrap_mean(constant, domains, n_resamples=200, seed=0)
    assert iv2.lo == pytest.approx(iv2.hi, abs=1e-12)


def test_sign_flip_is_exact_for_the_planned_design_size():
    values = np.full(20, 0.03)
    res = sign_flip_test(values)
    assert res.exact and res.n_permutations == 2**20
    assert res.p_one_sided == pytest.approx(1 / 2**20, abs=1e-9)


def test_sign_flip_symmetric_data_gives_uninformative_p():
    """Symmetric contrasts give an uninformative p, biased upward by ties.

    With an observed mean of exactly zero, every permutation whose mean is also
    exactly zero counts toward the one-sided tail. Here 16 of 64 permutations sit
    on that atom, so p lands at 0.625 rather than 0.5. Conservative, and correct
    for a discrete randomization test -- the test must never report a p below the
    truth.
    """
    values = np.array([0.1, -0.1, 0.05, -0.05, 0.0, 0.0])
    res = sign_flip_test(values)
    assert res.statistic == pytest.approx(0.0)
    assert 0.5 <= res.p_one_sided <= 0.7


def test_sign_flip_falls_back_to_monte_carlo_above_threshold():
    res = sign_flip_test(np.full(25, 0.02), n_mc=2000, seed=0)
    assert not res.exact and res.n_permutations == 2000


def test_agreement_rule_has_three_explicit_outcomes():
    positive_iv = Interval(0.1, 0.01, 0.2, 0.02, 20)
    negative_iv = Interval(0.01, -0.1, 0.1, -0.08, 20)
    positive_sf = SignFlipResult(0.1, 0.01, True, 2**20)
    negative_sf = SignFlipResult(0.01, 0.40, True, 2**20)

    assert (
        agreement_decision(positive_iv, positive_sf).status
        is DecisionStatus.FLAG
    )
    assert (
        agreement_decision(negative_iv, negative_sf).status
        is DecisionStatus.NOT_FLAGGED
    )
    assert (
        agreement_decision(positive_iv, negative_sf).status
        is DecisionStatus.INCONCLUSIVE
    )
    assert (
        agreement_decision(negative_iv, positive_sf).status
        is DecisionStatus.INCONCLUSIVE
    )


def test_placebo_rank_is_a_rank_not_a_p_value():
    assert placebo_rank(0.20, [0.05, -0.02]) == "1/3"
    assert placebo_rank(0.01, [0.05, -0.02]) == "2/3"
    assert placebo_rank(-0.10, [0.05, -0.02]) == "3/3"


def test_leave_one_domain_out_drops_exactly_one_domain():
    domains = np.array(make_domains(20, 5))
    values = np.where(domains == "domain_0", 1.0, 0.0)
    lodo = leave_one_domain_out(values, domains)
    assert len(lodo) == 5
    assert lodo["domain_0"] == pytest.approx(0.0)
    assert lodo["domain_1"] == pytest.approx(4 / 16)


def test_bootstrap_rejects_empty_and_misaligned_input():
    with pytest.raises(ValueError):
        bootstrap_mean(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        bootstrap_mean(np.zeros(3), np.array(["a", "b"]))
