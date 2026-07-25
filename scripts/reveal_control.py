#!/usr/bin/env python3
"""Reveal the byte-identical control only after decisions have been sealed."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.blinding import (  # noqa: E402
    identify_exact_control,
    require_frozen_decisions,
    sha256_file,
)


def parse_organism(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("organism must be LABEL=/absolute/model/path")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--organism", type=parse_organism, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing reveal: {args.output}")
    require_frozen_decisions(args.decision_manifest)
    organisms = dict(args.organism)
    if len(organisms) != len(args.organism):
        raise ValueError("organism labels must be unique")
    control, manifests = identify_exact_control(args.base, organisms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private_manifests = _write_private_manifests(args.output, manifests)
    payload = {
        "kind": "concealed_label_control_reveal",
        "revealed_at_utc": datetime.now(UTC).isoformat(),
        "decision_manifest_sha256": sha256_file(args.decision_manifest),
        "control_label": control,
        "weight_manifest_sha256": {
            label: sha256_file(path)
            for label, path in private_manifests.items()
        },
    }
    with args.output.open("x") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": "REVEALED", "control_label": control}))
    return 0


def _write_private_manifests(
    output_path: Path, manifests: dict[str, list[dict[str, object]]]
) -> dict[str, Path]:
    """Write per-label weight hashes without placing local model paths in output."""
    directory = output_path.parent / f"{output_path.stem}_weight_manifests"
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for label, manifest in manifests.items():
        path = directory / f"{label}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        paths[label] = path
    return paths


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
