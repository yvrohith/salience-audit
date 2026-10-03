"""Q1 power expectations from v1 data (Phase A; uses only v1 pool runs, never v2 outputs).

v1 informed the choice of statistic, so these numbers are optimistic by construction:
Macron's v1 residual was noticed after the fact (winner's curse). We therefore report
power under the observed effect AND under shrunken effects, and how concentrated the v1
effect is across templates (a template-specific effect may not transfer to fresh templates).

Simulation: draw 30 templates with replacement from v1's 20 (domain labels ignored),
recompute the v2 statistic (leave-one-principal-out flattening residual) for organism A,
and record whether Macron ranks <= 2 of 46. Shrinkage f moves Macron's effect a fraction f
of the way to the pool median residual (not to 0: A's residuals are not centred on 0).

    uv run python analysis/v2/power_v2.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("analyze_v2", ROOT / "scripts" / "analyze_v2.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
OUT = Path(__file__).with_name("power_v2.json")
N_SIM = 4000
SEED = 20261007


def load(name: str) -> dict:
    return {(r["principal_id"], r["template_id"], r["order"]): r["p_principal"]
            for r in map(json.loads, open(ROOT / "runs" / "pool" / f"{name}.jsonl"))}


def main() -> None:
    base, org = load("base"), load("organism_a")
    keys = sorted(base)
    ids = sorted({k[0] for k in keys})
    temps = sorted({k[1] for k in keys})
    mac = ids.index("emmanuel_macron")
    pb = np.array([base[k] for k in keys])
    po = np.array([org[k] for k in keys])
    pidx = np.array([ids.index(k[0]) for k in keys])
    tidx = np.array([temps.index(k[1]) for k in keys])
    full = A.loo_residuals(pb, po, pidx, len(ids))
    r_mac = float(full["residual"][mac])
    lam = float(full["lambda_loo"][mac])

    # Per-template contribution to Macron's residual (both orders averaged).
    per_t = []
    for t in range(len(temps)):
        sel = (pidx == mac) & (tidx == t)
        per_t.append(float(np.mean((po[sel] - pb[sel]) - lam * (0.5 - pb[sel]))))
    per_t = np.array(per_t)
    others = []
    for k in range(len(ids)):
        if k == mac:
            continue
        lk = float(full["lambda_loo"][k])
        for t in range(len(temps)):
            sel = (pidx == k) & (tidx == t)
            others.append(float(np.mean((po[sel] - pb[sel]) - lk * (0.5 - pb[sel]))))
    others = np.array(others)

    rng = np.random.default_rng(SEED)
    cell_rows = {t: np.flatnonzero(tidx == t) for t in range(len(temps))}
    # Shrink toward the pool MEDIAN residual of the other principals (A's residuals are not
    # centred on 0; anchoring at 0 overstates power - red-team finding 7).
    anchor = float(np.median(np.delete(full["residual"], mac)))
    power = {}
    for f in (0.0, 0.25, 0.375, 0.5, 0.75, 1.0):
        po_f = po.copy()
        po_f[pidx == mac] -= f * (r_mac - anchor)
        hits = 0
        ranks = []
        for _ in range(N_SIM):
            draw = rng.integers(0, len(temps), size=30)
            rows = np.concatenate([cell_rows[t] for t in draw])
            r = A.loo_residuals(pb[rows], po_f[rows], pidx[rows], len(ids))["residual"]
            rk = A.rank_of(r, mac)
            ranks.append(rk)
            hits += rk <= 2
        power[f"shrink_{f}"] = {"effect_fraction_kept": 1 - f,
                                "macron_residual_v1_scaled": anchor + (1 - f) * (r_mac - anchor),
                                "power_rank_le_2": hits / N_SIM, "median_rank": float(np.median(ranks))}
    sorted_t = np.sort(per_t)[::-1]
    out = {
        "status": "Phase A power expectation from v1 organism-A pool data (exploratory; optimistic)",
        "v1_macron_residual": r_mac,
        "anchor_pool_median_residual": anchor,
        "uniform_prior_power": None,
        "v1_macron_rank": A.rank_of(full["residual"], mac),
        "v1_macron_robust_z": A.robust_z(full["residual"], mac),
        "per_template_macron_residual": dict(zip(temps, per_t.tolist())),
        "n_templates_macron_positive": int(np.sum(per_t > 0)),
        "share_of_macron_effect_in_top3_templates": float(sorted_t[:3].sum() / per_t.sum()) if per_t.sum() > 0 else None,
        "other_principals_per_template_sd": float(np.std(others, ddof=1)),
        "macron_per_template_sd": float(np.std(per_t, ddof=1)),
        "power": power,
        "n_sim": N_SIM,
    }
    ps = [v["power_rank_le_2"] for v in power.values()]
    fr = [v["effect_fraction_kept"] for v in power.values()]
    order = np.argsort(fr)
    out["uniform_prior_power"] = float(np.trapezoid(np.array(ps)[order], np.array(fr)[order]))
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_template_macron_residual"}, indent=1))


if __name__ == "__main__":
    main()
