"""Q5b positive control: analyse scored runs with the FROZEN v2 statistic (imported, not reimplemented).

For each available seed k (runs/v2/pool/q5b_dahl_s{k}.jsonl, paired with lora_f2_s{k}):
  - Dahl's rank, residual and one-sided p under the Q1 rule (leave-one-principal-out flattening
    residual; rank <= 2 of 46);
  - Dahl's residual rise against the paired clean LoRA, against the median rise of all 46;
  - share of the 46 residuals that rose; mean rise of the 45 non-Dahl principals;
  - lambda with its template-bootstrap CI; the readout validity gate (median mass >= 0.5).
Predictions (discovery/q5b_addendum.json): primary Dahl rank <= 2 in seed 0; secondary Dahl's rise
beats the median rise in seed 0. Seeds 1-2 are replications, reported descriptively.

    uv run python scripts/q5b_analyze.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_v2", ROOT / "scripts" / "analyze_v2.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

POOL_DIR = ROOT / "runs" / "v2" / "pool"
OUT = ROOT / "analysis" / "v2" / "q5b" / "q5b_results.json"
TARGET = "henrik_dahl"


def main() -> None:
    manifest = A.MANIFEST
    freeze = A._load_module("freeze_v2", "scripts/freeze_v2.py")
    freeze.verify_v2_freeze(manifest)
    msha = freeze.sha256_file(manifest)
    roster = json.loads(manifest.read_text())["checkpoint_roster"]
    thr, min_mass = int(roster["rank_threshold"]), float(roster["validity_min_median_mass"])
    ids = sorted(A.load_pool(A.POOL).ids())
    base = A.Checkpoint("base", A.load_jsonl(POOL_DIR / "base.jsonl"))
    keys = base.keys
    pidx = np.array([ids.index(k[0]) for k in keys])
    temps = sorted({k[1] for k in keys})
    tidx = np.array([temps.index(k[1]) for k in keys])
    tdom = np.array([base.domain_of[t] for t in temps])
    ti = ids.index(TARGET)

    def load(name: str):
        path = POOL_DIR / f"{name}.jsonl"
        if not path.exists():
            return None
        why = A.provenance(path, name, msha)
        if why:
            raise SystemExit(f"{name}: {why}")
        return A.Checkpoint(name, A.load_jsonl(path), keys)

    out = {"status": "Q5b positive control (addendum to the frozen v2 study; descriptive except the two predictions)",
           "statistic": "frozen v2 Q1 statistic imported from scripts/analyze_v2.py", "target": TARGET,
           "rank_threshold": thr, "seeds": {}}
    q5 = load("q5_dahl")
    if q5 is not None:
        r5 = A.loo_residuals(base.p, q5.p, pidx, len(ids))["residual"]
        out["reference_q5_dahl"] = A.q_rule(r5, ids, TARGET, upper=True, threshold=thr, label="rank_le_threshold")
    for k in (0, 1, 2):
        ck, paired = load(f"q5b_dahl_s{k}"), load(f"lora_f2_s{k}")
        if ck is None or paired is None:
            continue
        r = A.loo_residuals(base.p, ck.p, pidx, len(ids))["residual"]
        rp = A.loo_residuals(base.p, paired.p, pidx, len(ids))["residual"]
        rise = r - rp
        others = np.delete(rise, ti)
        rule = A.q_rule(r, ids, TARGET, upper=True, threshold=thr)
        out["seeds"][f"s{k}"] = {
            "checkpoint": f"q5b_dahl_s{k}", "paired_clean": f"lora_f2_s{k}",
            **A.validity(ck, min_mass),
            "dahl": rule,
            "dahl_rise": float(rise[ti]), "median_rise": float(np.median(rise)),
            "dahl_rise_rank": A.rank_of(rise, ti), "dahl_rise_beats_median": bool(rise[ti] > np.median(rise)),
            "share_residuals_rose": float(np.mean(rise > 0)),
            "mean_rise_non_dahl": float(others.mean()),
            "lambda": A.lambda_ci(base.p, ck.p, tidx, tdom),
            "top3_residual": [[ids[j], float(r[j])] for j in np.argsort(-r)[:3]],
            "residual": {pid: float(r[i]) for i, pid in enumerate(ids)},
        }
    s0 = out["seeds"].get("s0")
    out["predictions"] = {
        "primary_dahl_rank_le_2_seed0": None if s0 is None else s0["dahl"]["significant"],
        "secondary_dahl_rise_beats_median_seed0": None if s0 is None else s0["dahl_rise_beats_median"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"predictions": out["predictions"],
                      "seeds": {k: {"rank": v["dahl"]["rank"], "residual": round(v["dahl"]["residual"], 4),
                                    "p": round(v["dahl"]["p_one_sided"], 4), "rise": round(v["dahl_rise"], 4),
                                    "median_rise": round(v["median_rise"], 4), "share_rose": v["share_residuals_rose"],
                                    "lambda": round(v["lambda"]["lambda"], 3), "valid": v["valid"]}
                                for k, v in out["seeds"].items()}}, indent=1))


if __name__ == "__main__":
    main()
