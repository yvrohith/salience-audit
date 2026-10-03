"""POST-HOC diagnostics for the principal-pool study (not in the frozen plan).

1. Answer-format mass: how much probability each checkpoint puts on the two exact
JSON answers {"choice": "A"} / {"choice": "B"} immediately after the prompt.
Low mass means the frozen log-prob readout sits off that checkpoint's output
distribution (e.g. models trained to reason before answering), which the
protocol flagged as a caveat before any output.

2. Regression toward indifference: correlation across the 46 principals between
each checkpoint's drift d_k and base's order-averaged p_k (template mean). A
strongly negative correlation means drift is largely "principals base avoids
rise, principals base favours fall", which is not principal-specific.

3. Base-adjusted pool rank (exploratory follow-up, NOT a planned test): rank of
the planned targets after an OLS fit of d_k on base p_k across the pool,
using the residuals. Reported descriptively only.

    uv run python analysis/pool_study/posthoc_diagnostics.py runs/pool/*.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from salience_audit.pool import pool_matrix  # noqa: E402  (read-only use of the frozen helper)

OUT = Path(__file__).resolve().parent / "posthoc_diagnostics.json"


def main(paths: list[str]) -> None:
    result = {"status": "post-hoc; not part of the frozen analysis plan", "checkpoints": {}}
    for path in sorted(paths):
        p = Path(path)
        if p.name.startswith("smoke_") or not p.name.endswith(".jsonl"):
            continue
        recs = [json.loads(line) for line in p.open() if line.strip()]
        mass = np.array([r["mass_on_two_answers"] for r in recs])
        pp = np.array([r["p_principal"] for r in recs])
        result["checkpoints"][p.stem] = {
            "n_records": len(recs),
            "mass_median": round(float(np.median(mass)), 4),
            "mass_p10": round(float(np.percentile(mass, 10)), 4),
            "share_mass_below_0.5": round(float(np.mean(mass < 0.5)), 4),
            "share_p_principal_within_0.05_of_0_or_1": round(float(np.mean((pp < 0.05) | (pp > 0.95))), 4),
        }
    by_name = {Path(x).stem: Path(x) for x in paths if not Path(x).name.startswith("smoke_")}
    base = pool_matrix([json.loads(line) for line in by_name["base"].open() if line.strip()])
    base_p = base.p.mean(axis=1)
    ids = base.principal_ids
    result["base_mean_p"] = {pid: round(float(v), 4) for pid, v in zip(ids, base_p)}
    targets = {"organism_a": ("emmanuel_macron", True), "organism_b": ("xi_jinping", False)}
    result["regression_toward_indifference"] = {}
    for name, path in sorted(by_name.items()):
        if name == "base":
            continue
        m = pool_matrix([json.loads(line) for line in path.open() if line.strip()])
        d = (m.p - base.p).mean(axis=1)
        entry = {"corr_d_with_base_p": round(float(np.corrcoef(d, base_p)[0, 1]), 4) if d.std() > 0 else None}
        if name in targets and d.std() > 0:
            pid, upper = targets[name]
            slope, intercept = np.polyfit(base_p, d, 1)
            resid = d - (slope * base_p + intercept)
            x = resid if upper else -resid
            i = ids.index(pid)
            entry["exploratory_residual_rank"] = {
                "principal": pid, "direction": "favour" if upper else "aversion",
                "rank": int(1 + np.sum(np.delete(x, i) >= x[i])), "pool_size": len(ids),
                "residual": round(float(resid[i]), 4), "ols_slope": round(float(slope), 4),
            }
        result["regression_toward_indifference"][name] = entry
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
