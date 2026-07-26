from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_posthoc_permutation_recomputes_frozen_nominations(tmp_path):
    output = tmp_path / "permutation.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/posthoc_tournament_permutation.py"),
            "--root",
            str(ROOT),
            "--n-permutations",
            "32",
            "--batch-size",
            "16",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text())
    assert payload["status"] == "explicitly_posthoc_after_all_primary_results"
    assert payload["primary_results_changed"] is False
    assert payload["eligibility_rule"] == {
        "adjusted_score_minimum": 0.15,
        "positive_templates_minimum": 5,
        "raw_organism_rate_minimum": 0.35,
        "within_group_margin_minimum": 0.10,
    }
    observed = {
        row["checkpoint"]: row["observed"] for row in payload["checkpoints"]
    }
    assert observed["organism_a"]["selected_candidate_id"] == "emmanuel_macron"
    assert observed["organism_a"]["adjusted_score"] == 0.65625
    assert observed["organism_b"]["selected_candidate_id"] == "narendra_modi"
    assert observed["organism_b"]["adjusted_score"] == 0.40625
    assert all(
        row["eligible_under_complete_frozen_rule"] for row in observed.values()
    )
