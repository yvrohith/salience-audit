"""Phase A dose pilot -> learning rates, ladder multipliers and cap (weights only).

Reads the merged-norm curves of the pilot LoRAs (runs/v2/train/pilot_*/norms.json; trained on
the disjoint UltraChat test split, never scored) and applies the rule frozen in
PROTOCOL_V2.md section 6:

  1. Fit norm(t) = a + b * ln(t) to each pilot curve (t = training step).
  2. LR exponent alpha: the pilot shows norm(t, lr) ~ norm(t, lr0) * (lr / lr0)^alpha. Estimate
     alpha from the two F2 pilots run at different learning rates (mean log-ratio over shared steps).
  3. Per family, choose LR so the predicted curve, rescaled by (LR / lr_pilot)^alpha, reaches the
     target at T_STAR = 300 steps. Round to 3 significant figures.
     F2: the curve is the log fit of the 3,000-example LR 1e-4 pilot.
     F1: the F1 pilot has only 535 examples (one epoch = 134 steps), so its later saves are
     inflated by memorisation. Its LEVEL is taken from its first save (step 100, single epoch)
     and its SHAPE from the F2 log fit: F1(t) = F2fit(t) * F1(100) / F2fit_at_F1_lr(100).
     This predicts lower F1 norms than the raw F1 curve, i.e. errs toward reaching the target.
  4. Ladder multipliers m = dose^(1/alpha) for doses 0.5 and 2.0, so ladder runs reach their dose
     at about the same step.
  5. Cap = 2,000 iterations (about 6.7 x T_STAR; covers real curves up to ~25% below the pilot's).

    uv run python analysis/v2/dose_pilot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "runs" / "v2" / "train"
OUT = Path(__file__).with_name("dose_pilot.json")
T_STAR = 300
CAP = 2000
PILOTS = {  # run dir -> (family, learning rate, data)
    "pilot_F2_3k": ("F2", 1.0e-4, "test_sft 3,000 examples (F2)"),
    "pilot_F2_lr15": ("F2", 1.5e-4, "test_sft 3,000 examples (F2)"),
    "pilot_F1_lr135": ("F1", 1.35e-4, "test_sft 535 self-distilled examples (F1)"),
}


def sig3(x: float) -> float:
    return float(f"{x:.3g}")


def main() -> None:
    target = yaml.safe_load((ROOT / "templates" / "lora_config_v2.yaml").read_text())["norm_rule"]["target"]
    curves = {}
    for run, (fam, lr, data) in PILOTS.items():
        pts = json.loads((TRAIN / run / "norms.json").read_text())
        t = np.array([p["step"] for p in pts], float)
        n = np.array([p["norm"] for p in pts], float)
        b, a = np.polyfit(np.log(t), n, 1)
        curves[run] = {"family": fam, "lr": lr, "data": data, "steps": t.tolist(), "norms": n.tolist(),
                       "fit_a": float(a), "fit_b": float(b),
                       "per_module_last": pts[-1].get("per_module")}
    lo, hi = curves["pilot_F2_3k"], curves["pilot_F2_lr15"]
    shared = sorted(set(lo["steps"]) & set(hi["steps"]))
    ratios = [hi["norms"][hi["steps"].index(s)] / lo["norms"][lo["steps"].index(s)] for s in shared]
    alpha = float(np.mean([math.log(r) for r in ratios]) / math.log(hi["lr"] / lo["lr"]))
    f2 = curves["pilot_F2_3k"]
    def f2fit(t: float) -> float:   # F2 log fit at its pilot LR (1e-4)
        return f2["fit_a"] + f2["fit_b"] * math.log(t)
    f1 = curves["pilot_F1_lr135"]
    f1_level = f1["norms"][f1["steps"].index(100.0)] / (f2fit(100) * (f1["lr"] / f2["lr"]) ** alpha)
    shape = {"F2": (lambda t: f2fit(t)), "F1": (lambda t: f2fit(t) * f1_level)}   # both at LR f2["lr"]
    chosen = {}
    for fam in ("F2", "F1"):
        pred = shape[fam](T_STAR)
        lr = sig3(f2["lr"] * (target / pred) ** (1 / alpha))
        scale = (lr / f2["lr"]) ** alpha
        chosen[fam] = {"learning_rate": lr, "curve_at_lr_1e-4_t_star": pred,
                       "predicted_norm_at_steps": {str(s): shape[fam](s) * scale
                                                   for s in (100, 200, 300, 400, 600, 1000, 2000)}}
    chosen["F1"]["level_vs_F2_from_first_save"] = f1_level
    mult = {str(d): sig3(d ** (1 / alpha)) for d in (0.5, 2.0)}
    out = {"status": "Phase A dose calibration (weights only; disjoint split; pilot adapters never scored)",
           "target": target, "t_star": T_STAR, "iters_cap": CAP,
           "alpha_lr_exponent": alpha, "alpha_from_ratios": dict(zip(map(str, shared), ratios)),
           "curves": curves, "chosen": chosen, "ladder_lr_multipliers": mult,
           "caveats": ["F1 pilot used 535 examples (multi-epoch after ~134 steps); real F1 runs use ~2,000",
                       "alpha estimated on F2 only and assumed for F1",
                       "pilot ran partly concurrently with generation; speeds in RUNLOG"]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("alpha_lr_exponent", "chosen", "ladder_lr_multipliers", "iters_cap")}, indent=1))


if __name__ == "__main__":
    main()
