"""Compact three-panel v2 figure for the revised report (Figure 4).

Plots only numbers already in analysis/v2/results/v2_results.json (Q1, Q3, Q4); no new analysis.

    uv run python analysis/report_figures/make_fig4_v2.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "analysis/v2/results/v2_results.json").read_text())
ck = d["checkpoints"]
q3 = d["Q3"]["by_checkpoint"]
q4 = d["Q4"]
CLEAN = ["lora_f1_s0", "lora_f1_s1", "lora_f1_s2", "lora_f2_s0", "lora_f2_s1", "lora_f2_s2",
         "benign_webshop", "benign_reasonrank", "benign_elyza",
         "ladder_f1_s0_x0p5", "ladder_f1_s0_x2", "ladder_f2_s0_x0p5", "ladder_f2_s0_x2"]
SHORT = {"organism_a": "A", "organism_b": "B", "lora_f1_s0": "F1 s0", "lora_f1_s1": "F1 s1", "lora_f1_s2": "F1 s2",
         "lora_f2_s0": "F2 s0", "lora_f2_s1": "F2 s1", "lora_f2_s2": "F2 s2", "benign_webshop": "WebShop",
         "benign_reasonrank": "ReasonRank", "benign_elyza": "ELYZA", "ladder_f1_s0_x0p5": "F1 ×0.5",
         "ladder_f1_s0_x2": "F1 ×2", "ladder_f2_s0_x0p5": "F2 ×0.5", "ladder_f2_s0_x2": "F2 ×2"}
RED, DARK, MID, LIGHT, GREY = "#c0392b", "#1f4e79", "#2e86c1", "#85c1e9", "#9aa1ab"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1.0, 1.05, 1.15]})

# (a) Q1: A's residuals, sorted
res = ck["organism_a"]["residual"]
order = sorted(res, key=res.get, reverse=True)
a.bar(range(1, len(order) + 1), [res[p] for p in order], width=0.8,
      color=[RED if p == "emmanuel_macron" else GREY for p in order])
clean_max = max(v["value"] for v in q4["max_residual_per_clean_control"].values())
a.axhline(clean_max, ls="--", lw=0.7, color="k")
a.text(46, clean_max + 0.006, "largest residual in\nany clean control", ha="right", va="bottom", fontsize=6)
a.text(2.2, res["emmanuel_macron"], "Macron", color=RED, va="center", fontsize=6.5)
a.axhline(0, lw=0.5, color="k")
a.set_xlabel("principal rank (1 = highest)")
a.set_ylabel("flattening-adjusted residual")
a.set_title("(a) Q1: organism A, 46 principals")
a.set_xlim(0, 47)
a.set_ylim(-0.2, 0.23)

# (b) Q3: lambda with CI
rows = ["organism_a", "organism_b"] + CLEAN
col = {r: (RED if r.startswith("organism") else DARK if r.startswith("lora") else MID if r.startswith("benign") else LIGHT)
       for r in rows}
for i, r in enumerate(rows):
    lam, (lo, hi) = q3[r]["lambda"], q3[r]["ci95"]
    b.errorbar(lam, i, xerr=[[lam - lo], [hi - lam]], fmt="o", ms=3, color=col[r], elinewidth=0.8, capsize=1.5)
b.set_yticks(range(len(rows)), [SHORT[r] for r in rows], fontsize=6)
b.invert_yaxis()
b.set_xlim(0, 0.75)
b.set_xlabel("flattening λ (95% CI)")
b.set_title("(b) Q3: flattening by checkpoint")

# (c) Q4: Macron's residual and the largest residual in each clean control
x = range(len(CLEAN))
c.scatter(x, [q4["max_residual_per_clean_control"][r]["value"] for r in CLEAN], marker="^", s=14, color="#444",
          label="largest residual, any principal")
c.scatter(x, [ck[r]["residual"]["emmanuel_macron"] for r in CLEAN], s=12, color=RED, label="Macron's residual")
c.axhline(q4["organism_a_residual"], ls="--", lw=0.8, color=RED, label="organism A, Macron")
c.set_xticks(list(x), [SHORT[r] for r in CLEAN], rotation=60, ha="right", fontsize=6)
c.set_ylim(-0.02, 0.255)
c.set_ylabel("flattening-adjusted residual")
c.set_title("(c) Q4: 13 clean controls")
c.legend(fontsize=5.5, loc="upper left", frameon=False, borderaxespad=0.2, handletextpad=0.3)
fig.tight_layout(w_pad=0.8)
out = ROOT / "analysis/report_figures/fig4_v2.png"
fig.savefig(out, dpi=300)
print(out)
