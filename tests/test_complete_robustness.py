from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "complete_robustness.py"
SPEC = importlib.util.spec_from_file_location("complete_robustness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holm_adjust_known_example() -> None:
    adjusted = MODULE.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    raw = {"a": 0.049, "b": 0.001, "c": 0.02, "d": 0.9}
    adjusted = MODULE.holm_adjust(raw)
    ordered = sorted(raw, key=raw.get)
    assert all(
        adjusted[left] <= adjusted[right]
        for left, right in zip(ordered, ordered[1:])
    )
