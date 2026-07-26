#!/usr/bin/env python3
"""Post-reveal check that the exact control is byte-identical to the public base.

This script writes no sealed artifact. It recomputes the weight manifest of the
unmodified Qwen2.5-7B-Instruct base and shows that the
result is byte-identical to the control's manifest, whose SHA-256 was already
fixed inside ``control_reveal.json`` at reveal time. A reader who downloads the
public base weights can therefore confirm the byte-identity claim without
trusting this repository. When ``--output`` is supplied, it also writes the
recomputed post-reveal manifest to that path.

Usage:
    python scripts/verify_base_identity.py \
        --base models/Qwen2.5-7B-Instruct \
        --reveal artifacts/confirmation_results/control_reveal.json \
        --output artifacts/confirmation_results/base_weight_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _load_weight_manifest():
    """Load blinding.weight_manifest without importing the package.

    ``salience_audit/__init__.py`` pulls in the analysis stack and its third-party
    dependencies. This verifier is meant to run against a bare interpreter, so it
    loads the single module it needs directly. The logic still has one source of
    truth: ``src/salience_audit/blinding.py``.
    """
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "salience_audit" / "blinding.py"
    )
    spec = importlib.util.spec_from_file_location("_blinding", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.weight_manifest


weight_manifest = _load_weight_manifest()


def serialize(manifest: list[dict[str, object]]) -> str:
    """Match the byte format used for the sealed per-label manifests."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reveal = json.loads(args.reveal.read_text())
    control = reveal["control_label"]
    expected = reveal["weight_manifest_sha256"][control]

    text = serialize(weight_manifest(args.base))
    actual = hashlib.sha256(text.encode()).hexdigest()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)

    ok = actual == expected
    print(
        json.dumps(
            {
                "status": "MATCH" if ok else "MISMATCH",
                "control_label": control,
                "sealed_manifest_sha256": expected,
                "recomputed_base_manifest_sha256": actual,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
