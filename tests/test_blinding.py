from __future__ import annotations

import json

import pytest

from salience_audit.blinding import identify_exact_control, require_frozen_decisions


def _model(path, contents: list[bytes]):
    path.mkdir()
    for index, content in enumerate(contents, start=1):
        (path / f"model-{index:05d}-of-{len(contents):05d}.safetensors").write_bytes(
            content
        )
    return path


def test_identify_exact_control_uses_weight_bytes(tmp_path):
    base = _model(tmp_path / "base", [b"one", b"two"])
    organisms = {
        "A": _model(tmp_path / "a", [b"changed", b"two"]),
        "B": _model(tmp_path / "b", [b"one", b"two"]),
        "C": _model(tmp_path / "c", [b"one", b"changed"]),
    }
    label, manifests = identify_exact_control(base, organisms)
    assert label == "B"
    assert set(manifests) == {"A", "B", "C"}


def test_identify_exact_control_fails_on_ambiguous_label(tmp_path):
    base = _model(tmp_path / "base", [b"same"])
    organisms = {
        "A": _model(tmp_path / "a", [b"same"]),
        "B": _model(tmp_path / "b", [b"same"]),
    }
    with pytest.raises(ValueError, match="exactly one"):
        identify_exact_control(base, organisms)


def test_control_reveal_requires_complete_decision_seal(tmp_path):
    path = tmp_path / "decisions.json"
    with pytest.raises(ValueError, match="does not exist"):
        require_frozen_decisions(path)
    path.write_text(json.dumps({"kind": "concealed_label_decision_freeze"}))
    with pytest.raises(ValueError, match="missing fields"):
        require_frozen_decisions(path)
    path.write_text(
        json.dumps(
            {
                "kind": "concealed_label_decision_freeze",
                "frozen_at_utc": "2026-07-25T00:00:00Z",
                "protocol_commit": "abc123",
                "summary_sha256": "a" * 64,
                "decisions": {"A": {"counterbalanced": "not_flagged"}},
            }
        )
    )
    assert require_frozen_decisions(path)["protocol_commit"] == "abc123"
