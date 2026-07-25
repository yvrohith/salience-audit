"""Utilities for sealing decisions before revealing the exact clean control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_manifest(model_path: Path) -> list[dict[str, object]]:
    """Hash inference weights without reading model cards or printing labels."""
    model_path = Path(model_path)
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{model_path}: no safetensors weights found")
    return [
        {
            "name": shard.name,
            "size_bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
        for shard in shards
    ]


def identify_exact_control(
    base_model: Path, organisms: dict[str, Path]
) -> tuple[str, dict[str, list[dict[str, object]]]]:
    """Return the sole organism whose weight manifest exactly matches the base."""
    base = weight_manifest(base_model)
    manifests = {label: weight_manifest(path) for label, path in organisms.items()}
    matches = [label for label, manifest in manifests.items() if manifest == base]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one byte-identical control, found {len(matches)}"
        )
    return matches[0], manifests


def require_frozen_decisions(path: Path) -> dict[str, object]:
    """Validate the seal required before any control comparison is performed."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"decision manifest does not exist: {path}")
    payload = json.loads(path.read_text())
    required = {
        "kind",
        "frozen_at_utc",
        "protocol_commit",
        "summary_sha256",
        "decisions",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"decision manifest missing fields: {missing}")
    if payload["kind"] != "concealed_label_decision_freeze":
        raise ValueError("not a concealed-label decision manifest")
    if not payload["decisions"]:
        raise ValueError("decision manifest contains no decisions")
    return payload
