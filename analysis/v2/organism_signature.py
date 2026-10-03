"""Weights-only fine-tune signature of organisms A and B against base (v2 Phase A, step A2).

For every tensor: whether it changed. For every changed matrix: the Frobenius norm of
dW = W_org - W_base (computed in float64 from the stored bf16 values) and its full
singular-value spectrum. A rank-r LoRA merged into bf16 weights shows r large singular
values followed by a flat rounding floor; r is read off the largest ratio
s_i / s_(i+1).

No model is run. Output: analysis/v2/organism_signature.json

    uv run python analysis/v2/organism_signature.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parents[1] / "models"
BASE = "Qwen2.5-7B-Instruct"
ORGANISMS = {"organism_a": "sl-organism-a-7b", "organism_b": "sl-organism-b-7b"}
OUT = Path(__file__).resolve().parent / "organism_signature.json"
TOP = 48  # singular values stored per matrix
GAP_SEARCH = 256  # look for the rank gap among the first this many values


class Shards:
    def __init__(self, dirname: str):
        self.dir = MODELS / dirname
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._cache: dict[str, dict] = {}

    def names(self) -> list[str]:
        return sorted(self.index)

    def get(self, name: str) -> np.ndarray:
        f = self.index[name]
        if f not in self._cache:
            self._cache[f] = mx.load(str(self.dir / f))
        return np.array(self._cache[f][name].astype(mx.float32), dtype=np.float64)


def module_key(name: str) -> tuple[str, int]:
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
    return (m.group(2), int(m.group(1))) if m else (name, -1)


def main() -> None:
    base = Shards(BASE)
    names = base.names()
    out: dict = {"base": BASE, "method": __doc__.strip().splitlines()[0], "organisms": {}}
    deltas_for_cos: dict[str, dict[str, np.ndarray]] = {}
    for org, dirname in ORGANISMS.items():
        o = Shards(dirname)
        assert o.names() == names, "tensor sets differ"
        changed, per_matrix = [], {}
        keep = {}
        for n in names:
            wb, wo = base.get(n), o.get(n)
            if np.array_equal(wb, wo):
                continue
            d = wo - wb
            changed.append(n)
            rec = {"shape": list(d.shape), "frac_elements_changed": float(np.mean(d != 0)),
                   "fro": float(np.linalg.norm(d))}
            if d.ndim == 2:
                s = np.linalg.svd(d, compute_uv=False)
                k = min(GAP_SEARCH, len(s) - 1)
                ratios = s[:k] / np.maximum(s[1:k + 1], 1e-30)
                r = int(np.argmax(ratios)) + 1
                rec.update({
                    "sv_top": [float(x) for x in s[:TOP]],
                    "inferred_rank": r,
                    "gap_ratio": float(ratios[r - 1]),
                    "floor_median_sv": float(np.median(s)),
                    "share_fro2_in_top_r": float(np.sum(s[:r] ** 2) / np.sum(s ** 2)),
                })
                keep[n] = d.astype(np.float32)
            per_matrix[n] = rec
        deltas_for_cos[org] = keep
        mods: dict[str, dict] = {}
        for n, rec in per_matrix.items():
            mod, layer = module_key(n)
            m = mods.setdefault(mod, {"layers": [], "fro2": 0.0, "inferred_ranks": [], "gap_ratios": [],
                                      "share_top_r": []})
            m["layers"].append(layer)
            m["fro2"] += rec["fro"] ** 2
            if "inferred_rank" in rec:
                m["inferred_ranks"].append(rec["inferred_rank"])
                m["gap_ratios"].append(rec["gap_ratio"])
                m["share_top_r"].append(rec["share_fro2_in_top_r"])
        summary = {}
        for mod, m in sorted(mods.items()):
            ranks = m["inferred_ranks"]
            summary[mod] = {
                "n_layers": len(m["layers"]), "layers": f"{min(m['layers'])}..{max(m['layers'])}",
                "fro": math.sqrt(m["fro2"]),
                "inferred_rank_mode": max(set(ranks), key=ranks.count) if ranks else None,
                "inferred_rank_counts": {str(r): ranks.count(r) for r in sorted(set(ranks))},
                "gap_ratio_min": min(m["gap_ratios"]) if m["gap_ratios"] else None,
                "gap_ratio_median": float(np.median(m["gap_ratios"])) if m["gap_ratios"] else None,
                "share_fro2_in_top_r_median": float(np.median(m["share_top_r"])) if m["share_top_r"] else None,
            }
        total = math.sqrt(sum(r["fro"] ** 2 for r in per_matrix.values()))
        out["organisms"][org] = {
            "model_dir": dirname,
            "n_tensors": len(names),
            "n_changed": len(changed),
            "unchanged_examples": [n for n in names if n not in per_matrix][:6],
            "modules": summary,
            "total_fro": total,
            "per_matrix": per_matrix,
        }
        print(org, "changed", len(changed), "total ||dW||_F", round(total, 4))
        for mod, s in summary.items():
            print(f"  {mod:28s} layers {s['layers']:6s} fro {s['fro']:.4f} rank mode {s['inferred_rank_mode']} "
                  f"counts {s['inferred_rank_counts']} gap min {s['gap_ratio_min']:.1f} med {s['gap_ratio_median']:.1f}")
    a, b = (out["organisms"][k]["total_fro"] for k in ORGANISMS)
    out["norm_target_geometric_mean"] = math.sqrt(a * b)
    common = sorted(set(deltas_for_cos["organism_a"]) & set(deltas_for_cos["organism_b"]))
    dot = sum(float(np.sum(deltas_for_cos["organism_a"][n].astype(np.float64) * deltas_for_cos["organism_b"][n]))
              for n in common)
    out["cosine_dW_A_dW_B"] = dot / (a * b)
    print("target (geometric mean of totals):", round(out["norm_target_geometric_mean"], 4),
          "cos(dW_A, dW_B):", round(out["cosine_dW_A_dW_B"], 4))
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
