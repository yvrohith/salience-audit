"""Score v2 tournament runs against the sealed base run, reusing the sealed scorer unchanged.

`score_tournament` and `select_tournament_candidates` from src/salience_audit/tournament.py are
called exactly as in the sealed scripts/summarize_tournament.py; only the checkpoint set differs
(the sealed summariser hard-codes the four original checkpoints). Each checkpoint is scored
against base in its OWN call (identical results to a joint call, because every score is
checkpoint-vs-base), so one broken run cannot take down the others.

Frozen tournament validity gate (PROTOCOL_V2.md section 10): a checkpoint's tournament is
valid iff its run is complete with the frozen seed, at least 90% of its 384 responses parse
(status ok), and the sealed scorer completes (no template x candidate cell without a valid
response). Invalid runs are reported with the reason and excluded from Q2's denominator.

    uv run python scripts/summarize_tournament_v2.py --runs-dir runs/v2/tournament \
      --output runs/v2/tournament/summary_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from salience_audit.mlx_runner import sha256_file  # noqa: E402
from salience_audit.tournament import (  # noqa: E402
    TournamentCompletion,
    load_roster,
    load_tournament_templates,
    score_tournament,
    select_tournament_candidates,
)

MANIFEST = ROOT / "discovery" / "tournament_v2_freeze_manifest.json"
MIN_OK_SHARE = 0.90


def load_run(path: Path) -> list[TournamentCompletion]:
    """Same checks as the sealed summariser's load_run."""
    meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    if meta.get("kind") != "tournament_run":
        raise ValueError(f"{path}: not a tournament run")
    records = [TournamentCompletion.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != meta.get("n_requests"):
        raise ValueError(f"{path}: incomplete tournament run")
    if len({r.request_id for r in records}) != len(records):
        raise ValueError(f"{path}: duplicate request ids")
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "v2" / "tournament")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--output", type=Path, default=ROOT / "runs" / "v2" / "tournament" / "summary_v2.json")
    a = ap.parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    base_path = ROOT / manifest["base_reference"]["path"]
    if sha256_file(base_path) != manifest["base_reference"]["sha256"]:
        raise SystemExit("sealed base tournament run differs from the v2 freeze record")
    planned = [n for n in manifest["checkpoint_seeds"] if n != "base"]
    base_recs = load_run(base_path)
    groups = load_roster(ROOT / "templates" / "candidate_roster.yaml")
    templates = load_tournament_templates(ROOT / "templates" / "tournament_templates.yaml")
    run_sha = {"base.jsonl (sealed)": sha256_file(base_path)}
    status_counts = {"base": dict(Counter(r.status.value for r in base_recs))}
    t_status, selected, all_scores, missing = {}, {}, [], []
    for name in planned:
        p = a.runs_dir / f"{name}.jsonl"
        if not p.exists():
            missing.append(name)
            t_status[name] = {"valid": False, "reason": "tournament run missing"}
            continue
        try:
            recs = load_run(p)
            if {r.checkpoint for r in recs} != {name}:
                raise ValueError("checkpoint field differs from file name")
            meta = json.loads(p.with_suffix(p.suffix + ".meta.json").read_text())
            if meta.get("seed") != manifest["checkpoint_seeds"][name]:
                raise ValueError("seed differs from the v2 freeze")
            if meta.get("freeze_manifest_sha256") != sha256_file(a.manifest):
                raise ValueError("run was not produced under the v2 tournament freeze")
        except (ValueError, OSError) as exc:
            t_status[name] = {"valid": False, "reason": f"run rejected: {exc}"}
            continue
        run_sha[p.name] = sha256_file(p)
        counts = Counter(r.status.value for r in recs)
        status_counts[name] = dict(counts)
        ok_share = counts.get("ok", 0) / len(recs)
        if ok_share < MIN_OK_SHARE:
            t_status[name] = {"valid": False, "reason": f"ok share {ok_share:.3f} < {MIN_OK_SHARE}", "ok_share": ok_share}
            continue
        try:
            scores = score_tournament(base_recs + recs, groups, templates, base_checkpoint="base")
        except ValueError as exc:
            t_status[name] = {"valid": False, "reason": f"sealed scorer failed: {exc}", "ok_share": ok_share}
            continue
        t_status[name] = {"valid": True, "ok_share": ok_share}
        selected[name] = select_tournament_candidates(scores, [name])[name]
        all_scores.extend(scores)

    def row(s):
        return {"candidate_id": s.candidate_id, "candidate_name": s.candidate_name, "group_id": s.group_id,
                "adjusted_score": s.adjusted_score, "raw_rate": s.raw_rate, "base_rate": s.base_rate,
                "positive_templates": s.positive_templates, "group_margin": s.group_margin}

    payload = {
        "kind": "tournament_summary_v2",
        "manifest_sha256": sha256_file(a.manifest),
        "run_sha256": run_sha,
        "missing": missing,
        "tournament_status": t_status,
        "status_counts": status_counts,
        "selected": {c: (row(s) if s else None) for c, s in selected.items()},
        "scores": [{"checkpoint": s.checkpoint, **row(s), "eligible": s.eligible}
                   for s in sorted(all_scores, key=lambda s: (s.checkpoint, -s.adjusted_score, s.candidate_id))],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "missing": missing,
                      "invalid": {c: v["reason"] for c, v in t_status.items() if not v["valid"]},
                      "selected": {c: (v["candidate_id"] if v else None) for c, v in payload["selected"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
