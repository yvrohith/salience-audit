"""Public scored-data reproduction: statistical invariants and integrity checks."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "analysis" / "behavioral_validation"
spec = importlib.util.spec_from_file_location("behavioral_validation_reproduce", PACKAGE / "reproduce.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

@pytest.fixture(scope="module")
def released_data():
    return (R.read_json(PACKAGE / "data/metadata.json"),
            [json.loads(line) for line in (PACKAGE / "data/cells.jsonl").read_text().splitlines()])

def test_public_release_reproduces_every_gold_summary():
    metadata, cubes, _ = R.load_data(PACKAGE / "data")
    result = R.reproduce(metadata, cubes)
    assert R.compare_expected(result, R.read_json(PACKAGE / "data/expected_summaries.json")) > 150_000
    ties = R.target_tie_counts(result)
    for bank in R.BANKS:
        assert ties[bank]["seed1_target"]["canonical_argmax"] == 46
        assert ties[bank]["seed1_target"]["generated"] == 46
        assert not result["paired_readouts"]["banks"][bank]["generated_qualification"]["seed1"]["primary_engineering_criterion_pass"]

def test_manifest_rejects_changed_scored_data(tmp_path):
    destination = tmp_path / "data"
    shutil.copytree(PACKAGE / "data", destination)
    with (destination / "cells.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(R.DataError, match="Integrity check failed: cells.jsonl"):
        R.load_data(destination)

def test_incomplete_and_duplicate_grids_are_not_silently_averaged(released_data):
    metadata, records = released_data
    with pytest.raises(R.DataError, match="Incomplete scored-data grid"):
        R.build_cubes(metadata, records[:-1])
    with pytest.raises(R.DataError, match="Duplicate"):
        R.build_cubes(metadata, records + [records[-1]])

def test_invalid_generation_is_explicitly_outside_release_scope(released_data):
    metadata, records = released_data
    changed = list(records)
    changed[0] = {**records[0], "valid": [False, True], "g": [None, records[0]["g"][1]]}
    with pytest.raises(R.DataError, match="Missing generation is unsupported"):
        R.build_cubes(metadata, changed)

def test_inconsistent_canonical_winner_is_rejected(released_data):
    metadata, records = released_data
    index = next(i for i, row in enumerate(records) if row["p"][0] != .5 and row["c"][0] != "tie")
    changed = list(records);row = copy.deepcopy(records[index])
    row["c"][0] = "A" if row["c"][0] == "B" else "B";changed[index] = row
    with pytest.raises(R.DataError, match="Canonical winner contradicts"):
        R.build_cubes(metadata, changed)

def test_canonical_tie_has_fractional_point_and_ambiguity_bounds():
    model = {"p": np.array([[[.5, .7]]]), "mass": np.array([[[.8, .8]]]),
             "c": np.array([[["tie", "B"]]]), "g": np.array([[["A", "A"]]])}
    arrays = R.readout_arrays(model);canonical = arrays["canonical_argmax"]
    assert canonical["point"].tolist() == [[[.5, 1.]]]
    assert canonical["lower"].tolist() == [[[0., 1.]]]
    assert canonical["upper"].tolist() == [[[1., 1.]]]
    result = R.linear_summary({"c": canonical}, [("c", 1.)], [0], [1.], np.zeros((R.BOOTSTRAPS, 1), dtype=int))
    assert result["mean_complete_joint"] == .75
    assert result["conditional_point_ci95"] == [.75, .75]
    assert result["finite_bank_identification_bounds"] == [.5, 1.]
    assert result["missing_robust_bootstrap_ci95"] == [.5, 1.]
    assert arrays["generated"]["point"].mean() == .5
    counts = R.agreement_summary(model)
    assert (counts["canonical_tie_count"], counts["resolved_valid_pairs"], counts["disagree"], counts["agree"]) == (1, 1, 1, 0)

def test_conservative_ties_are_not_unique_last_place():
    levels = {"p" + str(i): {"mean_complete_joint": .5, "finite_bank_identification_bounds": [.5, .5]} for i in range(46)}
    assert all(s["conditional_point_rank"] == 46 for s in R.rank_levels(levels).values())
    assert all(s["finite_bank_identification_rank_bounds"] == [46, 46] for s in levels.values())
    ambiguous = {"p" + str(i): {"mean_complete_joint": .5, "finite_bank_identification_bounds": [0., 1.]} for i in range(46)}
    assert all(s["finite_bank_identification_rank_bounds"] == [1, 46] for s in R.rank_levels(ambiguous).values())

def test_generated_boundary_is_roundoff_stable():
    summary = {"validity_gate_pass": True, "mean_complete_joint": .19999999999999987, "missing_robust_bootstrap_ci95": [.05, .3]}
    assert R.generated_criterion(summary)
    assert not R.generated_criterion({**summary, "mean_complete_joint": .2 - 1 / 420})
    assert not R.generated_criterion({**summary, "missing_robust_bootstrap_ci95": [2.8e-17, .3]})
    assert R.generated_criterion({**summary, "missing_robust_bootstrap_ci95": [1 / (420 * 40), .3]})

def test_both_controls_required_even_when_target_is_strong():
    draws = R.bootstrap_indices(tuple("abcdef" * 2), R.BOOT_SEEDS["old"])
    def table(values):
        point = np.broadcast_to(np.array(values)[None, :, None], (12, 8, 2)).copy()
        return {"point": point, "lower": point.copy(), "upper": point.copy(), "tie": np.zeros_like(point, dtype=bool)}
    tables = {"target": table([1.] + [0.] * 7), "balanced": table([.5] * 8), "generic": table([1.] + [0.] * 7)}
    panel = list(range(8))
    assert R.generated_criterion(R.matched_summary(tables, [("target", 1)], panel, draws))
    assert R.generated_criterion(R.matched_summary(tables, [("target", 1), ("balanced", -1)], panel, draws))
    generic = R.matched_summary(tables, [("target", 1), ("generic", -1)], panel, draws)
    assert generic["mean_complete_joint"] == pytest.approx(0)
    assert not R.generated_criterion(generic)

def test_loo_fit_excludes_tested_principal():
    rng = np.random.default_rng(72);base = rng.uniform(.05, .95, size=(12, 4, 2));current = rng.uniform(.05, .95, size=base.shape)
    before = R.loo_points(base, current)
    changed = current.copy();changed[:, 2, :] = rng.uniform(0, 1, size=(12, 2));after = R.loo_points(base, changed)
    assert after["lambda_loo"][2] == pytest.approx(before["lambda_loo"][2], abs=1e-14)
    keep = [0, 1, 3];x = (.5 - base[:, keep, :]).ravel();d = (current[:, keep, :] - base[:, keep, :]).ravel()
    assert before["lambda_loo"][2] == pytest.approx(float(x @ d / (x @ x)), abs=1e-14)
    assert not np.allclose(after["flattening_adjusted_residual"], before["flattening_adjusted_residual"])

def test_uniform_flattening_removed_but_target_only_change_retained():
    rng = np.random.default_rng(17);base = rng.uniform(.02, .98, size=(12, 4, 2));lam = .37
    fitted = R.loo_points(base, .5 + (1 - lam) * (base - .5))
    assert np.allclose(fitted["lambda_loo"], lam, atol=1e-12)
    assert np.max(np.abs(fitted["flattening_adjusted_residual"])) < 1e-12
    planted = base.copy();planted[:, 1, :] += .2 * (1 - planted[:, 1, :]);recovered = R.loo_points(base, planted)
    assert recovered["lambda_loo"][1] == pytest.approx(0, abs=1e-12)
    assert recovered["flattening_adjusted_residual"][1] == pytest.approx(.2 * (1 - base[:, 1, :]).mean(), abs=1e-12)

def test_domain_bootstrap_pairs_within_banks_not_across_banks():
    domains = ("a", "b", "a", "b");old = R.bootstrap_indices(domains, R.BOOT_SEEDS["old"]);fresh = R.bootstrap_indices(domains, R.BOOT_SEEDS["fresh"])
    assert np.all((np.array(domains)[old] == "a").sum(1) == 2)
    assert np.all((np.array(domains)[old] == "b").sum(1) == 2)
    assert not np.array_equal(old, fresh)
    values = np.array([0., 0., 1., 1.]);vector = {"point": values, "lower": values, "upper": values}
    result = R.independent_banks(vector, vector, fresh, old)
    assert result["mean_complete_joint"] == 0
    assert result["conditional_point_ci95"][0] < 0 < result["conditional_point_ci95"][1]
    assert np.array_equal(values[old].mean(1) - values[old].mean(1), np.zeros(R.BOOTSTRAPS))

def test_gold_comparison_rejects_changed_metrics_counts_and_ranks():
    with pytest.raises(R.DataError, match="Numerical mismatch"):
        R.compare_expected({"effect": .31}, {"effect": .30})
    with pytest.raises(R.DataError, match="Value mismatch"):
        R.compare_expected({"rank": 45}, {"rank": 46})
    with pytest.raises(R.DataError, match="Value mismatch"):
        R.compare_expected({"count": 19}, {"count": 20})


def test_modified_scientific_configuration_is_not_silently_ignored(released_data):
    metadata, _ = released_data
    changed = copy.deepcopy(metadata)
    changed["analysis"]["strong_generated_criterion"]["point_minimum"] = .3
    with pytest.raises(R.DataError, match="criterion differs"):
        R.validate_metadata(changed)
    changed = copy.deepcopy(metadata)
    changed["analysis"]["bootstrap"]["shared_within_bank"] = False
    with pytest.raises(R.DataError, match="bootstrap method"):
        R.validate_metadata(changed)
