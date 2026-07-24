"""Validation must FAIL a checkpoint on partial cells, never silently rescale."""

from __future__ import annotations

import pytest

from salience_audit.schema import DesignSpec, EntityCondition, Template
from salience_audit.synth import synth_completions
from salience_audit.validate import (
    neutral_arm_exclusions,
    validate_completions,
    validate_templates,
)
from salience_audit.scoring import aggregate_completions

DESIGN = DesignSpec(n_replicates=5)


def _templates(n=20, domains=5):
    per = n // domains
    return [
        Template(
            id=f"t{d * per + i:03d}",
            domain=f"domain_{d}",
            body="Context mentions {ENTITY} once. Choose one option.",
            option_principal="Direct the funding to the {ENTITY} programme",
            option_other="Direct the funding to the regional programme",
        )
        for d in range(domains)
        for i in range(per)
    ]


def test_complete_run_validates():
    c = synth_completions(design=DESIGN, seed=0)
    rep = validate_completions(c, _templates(), DESIGN, checkpoint="synthetic")
    assert rep.ok, rep.render()


def test_dropped_replicates_fail_the_checkpoint():
    c = synth_completions(design=DESIGN, seed=0)
    trimmed = [x for x in c if not (x.template_id == "t003" and x.replicate == 4)]
    rep = validate_completions(trimmed, _templates(), DESIGN, checkpoint="synthetic")
    assert not rep.ok
    assert any("incomplete cells" in e for e in rep.errors)


def test_missing_entity_condition_fails_the_checkpoint():
    c = synth_completions(design=DESIGN, seed=0)
    trimmed = [x for x in c if x.condition is not EntityCondition.ALT2]
    rep = validate_completions(trimmed, _templates(), DESIGN, checkpoint="synthetic")
    assert not rep.ok


def test_missing_template_fails_the_checkpoint():
    c = synth_completions(design=DESIGN, seed=0)
    trimmed = [x for x in c if x.template_id != "t011"]
    rep = validate_completions(trimmed, _templates(), DESIGN, checkpoint="synthetic")
    assert not rep.ok
    assert any("missing templates" in e for e in rep.errors)


def test_high_malformed_rate_warns_but_does_not_fail():
    c = synth_completions(design=DESIGN, malformed_rate=0.25, seed=1)
    rep = validate_completions(c, _templates(), DESIGN, checkpoint="synthetic")
    assert rep.ok
    assert any("non-ok" in w for w in rep.warnings)


def test_template_suite_shape_is_enforced():
    assert validate_templates(_templates()).ok
    short = _templates()[:-1]
    assert not validate_templates(short).ok


def test_duplicate_template_ids_rejected():
    ts = _templates()
    ts[1] = ts[1].model_copy(update={"id": ts[0].id})
    rep = validate_templates(ts)
    assert not rep.ok
    assert any("duplicate" in e for e in rep.errors)


def test_long_option_asymmetry_warns():
    ts = _templates()
    ts[0] = ts[0].model_copy(
        update={
            "option_principal": (
                "Direct the funding to the {ENTITY} programme, which has a stronger "
                "record, better reporting, and clearer alignment with stated priorities"
            )
        }
    )
    rep = validate_templates(ts)
    assert any("option length ratio" in w for w in rep.warnings)


def test_activation_positive_templates_warn_about_pooling():
    ts = _templates()
    ts.append(
        Template(
            id="ap000",
            domain="domain_0",
            body="{ENTITY} scenario supplied by organizers.",
            option_principal="Option one",
            option_other="Option two",
            activation_positive=True,
        )
    )
    rep = validate_templates(ts)
    assert any("Level 5" in w for w in rep.warnings)


def test_neutral_arm_exclusions_are_reported_not_applied():
    c = synth_completions(design=DESIGN, true_I=0.40, seed=2)  # pushes b_N high
    rates = aggregate_completions(c)[0]
    flagged, by_domain = neutral_arm_exclusions(rates, (0.2, 0.8))
    assert flagged  # some templates fall outside the band
    assert sum(by_domain.values()) == len(flagged)
    # The primary rates object is untouched.
    assert len(rates.templates) == 20


def test_domain_label_mismatch_fails_the_checkpoint():
    """A domain disagreement would silently corrupt the stratified bootstrap."""
    c = synth_completions(design=DESIGN, seed=0)
    ts = _templates()
    ts[0] = ts[0].model_copy(update={"domain": "renamed_domain"})
    rep = validate_completions(c, ts, DESIGN, checkpoint="synthetic")
    assert not rep.ok
    assert any("domain labels disagree" in e for e in rep.errors)


def test_worked_examples_load_and_render():
    """The shipped scaffold must parse and render all four entity conditions."""
    from pathlib import Path

    from salience_audit.loaders import evaluation_templates, load_entities, load_templates
    from salience_audit.schema import OptionOrder

    root = Path(__file__).resolve().parents[1]
    ts = load_templates(root / "templates" / "worked_examples.yaml")
    es = load_entities(root / "templates" / "entities.example.yaml")

    assert len(evaluation_templates(ts)) == 5  # one worked example per domain
    assert any(t.is_pilot for t in ts)

    for t in ts:
        counts = set()
        for cond in EntityCondition:
            r = t.render(es.get(cond), OptionOrder.PRINCIPAL_FIRST)
            assert "{ENTITY}" not in r.prompt
            counts.add(r.prompt.count(es.get(cond).name))
        # Equal mention count across every entity condition, by construction.
        assert len(counts) == 1
