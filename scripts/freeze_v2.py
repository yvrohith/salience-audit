"""Freeze the v2 study before any v2 audit output exists.

Writes discovery/pool_v2_freeze_manifest.json, a manifest of kind `pool_freeze` that the
unchanged v1 runner (scripts/run_pool_logprob.py) accepts with
    --templates templates/frozen_suite_v2.yaml --freeze-manifest discovery/pool_v2_freeze_manifest.json
It records hashes of the v2 templates, the pool, every implementation file (v1 code reused
unchanged plus all v2 code), the protocol, keyword list, data recipe, LoRA config/norm rule,
checkpoint roster, the local Q5 overt-example file, and the tournament seeds. Run it before
scripts/freeze_tournament_v2.py, which checks its seeds against this manifest.
Refuses to overwrite, refuses if v2 pool or tournament outputs exist, refuses placeholders.

    uv run python scripts/freeze_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from salience_audit.loaders import evaluation_templates, load_templates  # noqa: E402
from salience_audit.pool import build_pool_requests, load_pool  # noqa: E402
from salience_audit.validate import validate_templates  # noqa: E402

TEMPLATES = ROOT / "templates" / "frozen_suite_v2.yaml"
POOL = ROOT / "templates" / "principal_pool.yaml"
PROTOCOL = ROOT / "analysis" / "v2" / "PROTOCOL_V2.md"
KEYWORDS = ROOT / "templates" / "political_keywords_v2.yaml"
RECIPE = ROOT / "templates" / "control_data_recipe_v2.yaml"
LORA = ROOT / "templates" / "lora_config_v2.yaml"
ROSTER = ROOT / "templates" / "checkpoint_roster_v2.yaml"


def tournament_seeds() -> dict[str, int]:
    """v2 tournament seeds, defined once here and re-derived/checked by freeze_tournament_v2.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ftv2", ROOT / "scripts" / "freeze_tournament_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.planned_seeds()

IMPLEMENTATION = [
    # v1 code reused unchanged (also hash-checked by run_pool_logprob.py at every run)
    "src/salience_audit/pool.py",
    "src/salience_audit/logprob.py",
    "src/salience_audit/schema.py",
    "src/salience_audit/loaders.py",
    "src/salience_audit/inference.py",
    "src/salience_audit/mlx_runner.py",
    "scripts/run_pool_logprob.py",
    "scripts/analyze_pool.py",
    # v2 code
    "scripts/freeze_v2.py",
    "scripts/freeze_tournament_v2.py",
    "scripts/build_controls_v2.py",
    "scripts/train_controls_v2.py",
    "scripts/analyze_v2.py",
    "scripts/summarize_tournament_v2.py",
    "tests/test_v2.py",
    "analysis/v2/organism_signature.py",
    "analysis/v2/organism_signature.json",
    "analysis/v2/power_v2.py",
    "analysis/v2/power_v2.json",
    "analysis/v2/dose_pilot.json",
    "src/salience_audit/validate.py",
    "src/salience_audit/tournament.py",
    "src/salience_audit/tournament_runner.py",
    "uv.lock",
]
PLACEHOLDER = ("TBD", "PLACEHOLDER")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "discovery" / "pool_v2_freeze_manifest.json")
    ap.add_argument("--allow-dirty-for-tests", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; a frozen manifest is never overwritten")
    for d in (ROOT / "runs" / "v2" / "pool", ROOT / "runs" / "v2" / "tournament"):
        if d.exists() and any(d.glob("*.jsonl")):
            raise SystemExit(f"{d} already contains outputs; freeze must precede them")
    lora = yaml.safe_load(LORA.read_text())
    if isinstance(lora["lora"]["iters_cap"], str) or any(isinstance(v, str) for v in lora["lora"]["learning_rate"].values()):
        raise SystemExit("LoRA config still has placeholders")
    for path in (PROTOCOL, LORA, RECIPE, ROSTER, KEYWORDS, TEMPLATES):
        hits = [w for w in PLACEHOLDER if w in path.read_text()]
        if hits:
            raise SystemExit(f"{path.name} still contains {hits}")
    # Everything frozen must already be committed: no modified tracked files, no untracked files.
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    study_dirs = ("analysis/", "templates/", "scripts/", "tests/", "discovery/", "src/")
    dirty = [line for line in status if not line.endswith(".DS_Store")
             and (not line.startswith("??") or line[3:].startswith(study_dirs))]
    if dirty and not a.allow_dirty_for_tests:
        raise SystemExit("commit all v2 files before freezing; git status shows:\n" + "\n".join(dirty))
    templates = load_templates(TEMPLATES)
    rep = validate_templates(templates, expected_domains=6, per_domain=5)
    if rep.errors:
        raise SystemExit(f"template validation failed: {rep.errors}")
    ev = evaluation_templates(templates)
    pool = load_pool(POOL)
    requests = build_pool_requests(ev, pool)
    recipe = yaml.safe_load(RECIPE.read_text())
    overt = ROOT / recipe["q5_positive_control"]["overt_examples"]
    if not overt.exists():
        raise SystemExit(f"Q5 overt examples missing: {overt}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "kind": "pool_freeze",
        "study": "v2 procedure-matched principal-pool study",
        "status": "Q1 confirmatory (single test); Q2-Q5 estimates/descriptive; see PROTOCOL_V2.md",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_status_at_freeze": dirty or "clean",
        "templates_path": str(TEMPLATES.relative_to(ROOT)),
        "templates_sha256": sha256_file(TEMPLATES),
        "pool_path": str(POOL.relative_to(ROOT)),
        "pool_sha256": sha256_file(POOL),
        "n_templates": len(ev),
        "n_principals": len(pool.principals),
        "n_requests_per_checkpoint": len(requests),
        "request_grid_sha256": hashlib.sha256(
            "\n".join(f"{r.request_id}\t{r.prompt_hash}" for r in requests).encode()).hexdigest(),
        "implementation_sha256": {p: sha256_file(ROOT / p) for p in IMPLEMENTATION},
        "protocol_sha256": sha256_file(PROTOCOL),
        "keyword_list_sha256": sha256_file(KEYWORDS),
        "data_recipe_sha256": sha256_file(RECIPE),
        "lora_config_sha256": sha256_file(LORA),
        "checkpoint_roster_sha256": sha256_file(ROSTER),
        "checkpoint_roster": yaml.safe_load(ROSTER.read_text()),
        "norm_rule": lora["norm_rule"],
        "lora": lora["lora"],
        "q5_overt_examples_sha256": sha256_file(overt),
        "tournament_seeds": tournament_seeds(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"froze {len(requests)} requests per checkpoint -> {a.output}")
    print(f"manifest sha256 {sha256_file(a.output)}")


def verify_v2_freeze(path: Path = ROOT / "discovery" / "pool_v2_freeze_manifest.json") -> dict:
    """Re-check every frozen hash; used before each Phase B step."""
    m = json.loads(Path(path).read_text())
    checks = {"templates_sha256": TEMPLATES, "pool_sha256": POOL, "protocol_sha256": PROTOCOL,
              "keyword_list_sha256": KEYWORDS, "data_recipe_sha256": RECIPE, "lora_config_sha256": LORA,
              "checkpoint_roster_sha256": ROSTER}
    bad = [k for k, p in checks.items() if sha256_file(p) != m[k]]
    bad += [rel for rel, d in m["implementation_sha256"].items() if sha256_file(ROOT / rel) != d]
    recipe = yaml.safe_load(RECIPE.read_text())
    if sha256_file(ROOT / recipe["q5_positive_control"]["overt_examples"]) != m["q5_overt_examples_sha256"]:
        bad.append("q5_overt_examples")
    if bad:
        raise SystemExit(f"v2 freeze verification failed: {bad}")
    return m


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_v2_freeze()
        print("v2 freeze verified")
    else:
        main()
