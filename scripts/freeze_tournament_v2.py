"""Freeze the v2 tournament runs for new checkpoints (sealed runner, roster, templates, rule unchanged).

Emits a `tournament_freeze` manifest that the sealed runner (scripts/run_tournament_mlx.py)
accepts: identical roster/template/implementation hashes to the sealed Stage 2 freeze,
plus NEW seeds for every v2 checkpoint that will be run. The base reference is the sealed
runs/tournament/base.jsonl; its hash is checked against the sealed summary and recorded.

    uv run python scripts/freeze_tournament_v2.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from salience_audit.mlx_runner import sha256_file  # noqa: E402
from salience_audit.tournament_runner import implementation_paths  # noqa: E402

SEALED_MANIFEST = ROOT / "discovery" / "tournament_freeze_manifest.json"
SEALED_SUMMARY = ROOT / "runs" / "tournament" / "summary.json"
SEALED_BASE = ROOT / "runs" / "tournament" / "base.jsonl"
ROSTER_V2 = ROOT / "templates" / "checkpoint_roster_v2.yaml"
SEED_BASE = 261003100  # v2 seeds: SEED_BASE + index among the NEW checkpoints in roster order
# (control_primary, control_ladder, positive); sealed seeds were 250720360-363.


def planned_seeds() -> dict[str, int]:
    roster = yaml.safe_load(ROSTER_V2.read_text())["checkpoints"]
    new = [e["name"] for e in roster if e["role"] in ("control_primary", "control_ladder", "positive")]
    return {name: SEED_BASE + i for i, name in enumerate(new)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "discovery" / "tournament_v2_freeze_manifest.json")
    ap.add_argument("--no-require-pool-freeze", dest="require_pool_freeze", action="store_false",
                    help="tests/dry checks only")
    a = ap.parse_args(argv)
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite existing freeze: {a.output}")
    out_dir = ROOT / "runs" / "v2" / "tournament"
    if out_dir.exists() and any(out_dir.glob("*.jsonl")):
        raise SystemExit(f"{out_dir} already contains tournament outputs; freeze must precede them")

    sealed = json.loads(SEALED_MANIFEST.read_text())
    impl = {rel: sha256_file(p) for rel, p in implementation_paths(ROOT).items()}
    if impl != sealed["implementation_sha256"]:
        raise SystemExit("tournament implementation differs from the sealed freeze; refusing")
    roster_path = ROOT / "templates" / "candidate_roster.yaml"
    templates_path = ROOT / "templates" / "tournament_templates.yaml"
    rule_path = ROOT / "discovery" / "TOURNAMENT_RULE.md"
    for field, path in (("roster_sha256", roster_path), ("templates_sha256", templates_path),
                        ("selection_rule_sha256", rule_path)):
        if sha256_file(path) != sealed[field]:
            raise SystemExit(f"{path.name} differs from the sealed freeze; refusing")
    summary = json.loads(SEALED_SUMMARY.read_text())
    base_sha = sha256_file(SEALED_BASE)
    if summary["run_sha256"]["base.jsonl"] != base_sha:
        raise SystemExit("sealed runs/tournament/base.jsonl differs from the sealed summary hash")

    seeds = planned_seeds()
    pool_v2 = ROOT / "discovery" / "pool_v2_freeze_manifest.json"
    if a.require_pool_freeze:
        if not pool_v2.exists():
            raise SystemExit("run scripts/freeze_v2.py first")
        if json.loads(pool_v2.read_text())["tournament_seeds"] != seeds:
            raise SystemExit("seeds differ from discovery/pool_v2_freeze_manifest.json")
    sealed_seeds = set(sealed["checkpoint_seeds"].values())
    if sealed_seeds & set(seeds.values()):
        raise SystemExit("seed collision with the sealed tournament")
    # Reproducibility check only (never a reference): re-run base with the SEALED seed under the
    # current software and compare raw outputs with the sealed base run.
    seeds_all = {"base": sealed["checkpoint_seeds"]["base"], **seeds}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    payload = {
        **{k: sealed[k] for k in ("roster_sha256", "templates_sha256", "selection_rule_sha256", "n_groups",
                                  "n_candidates", "n_templates", "n_rotations", "temperature", "max_tokens")},
        "kind": "tournament_freeze",
        "study": "v2",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_commit": commit,
        "implementation_sha256": impl,
        "checkpoint_seeds": seeds_all,
        "base_reproduction_check": {"checkpoint": "base", "seed": sealed["checkpoint_seeds"]["base"],
                                    "output": "runs/v2/tournament/base_repro.jsonl",
                                    "compare_to": "runs/tournament/base.jsonl"},
        "base_reference": {"path": "runs/tournament/base.jsonl", "sha256": base_sha,
                           "sealed_manifest": "discovery/tournament_freeze_manifest.json",
                           "sealed_seed": sealed["checkpoint_seeds"]["base"]},
        "summarizer": {"path": "scripts/summarize_tournament_v2.py",
                       "sha256": sha256_file(ROOT / "scripts" / "summarize_tournament_v2.py")},
        "checkpoint_roster_sha256": sha256_file(ROSTER_V2),
        "pool_v2_manifest_sha256": sha256_file(pool_v2) if pool_v2.exists() else None,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": "FROZEN", "output": str(a.output), "seeds": seeds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
