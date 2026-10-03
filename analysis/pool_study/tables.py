"""Formatting only: print the RESULTS.md tables straight from pool_summary.json.

No statistics are computed here beyond counting list lengths and comparing ranks
with the frozen rank threshold, so every number in the tables is traceable to
artifacts/pool_results/pool_summary.json.

    uv run python analysis/pool_study/tables.py [artifacts/pool_results/pool_summary.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(path: Path) -> None:
    s = json.loads(path.read_text())
    thr = s["rank_threshold"]
    print(f"pool_size={s['pool_size']} rank_threshold={thr}\n")

    v = s.get("validation_reference")
    print("## Validation (log-prob vs sampled, byte-identical prompts)\n")
    print("| checkpoint | cells | argmax = sampled majority | Pearson r | mean abs diff |\n|---|---:|---:|---:|---:|")
    rows = [("base", v)] + [(k, e.get("validation")) for k, e in s["organisms"].items()]
    for name, x in rows:
        if x:
            print(f"| {name} | {x['n_cells']} | {x['argmax_agrees_with_sampled_majority']:.4f} | "
                  f"{x['pearson_r']:.3f} | {x['mean_abs_diff']:.4f} |")

    print("\n## Planned tests\n")
    print("| id | checkpoint | principal | direction | d | rank | pool | p one-sided | Holm p | prediction | held |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for t in s["planned_tests"]:
        if t["principal"] == "all":
            print(f"| {t['id']} | {t['organism']} | all | both | max abs d = {t['max_abs_d']:.4f} | | | | | "
                  f"max abs d < 0.03 | {'yes' if t['pass'] else 'no'} |")
            continue
        sig = t["rank_from_top"] <= thr
        held = (not sig) if t["id"] == "H3" else sig
        print(f"| {t['id']} | {t['organism']} | {t['principal']} | {t['direction']} | {t['d']:+.4f} | "
              f"{t['rank_from_top']} | {t['pool_size']} | {t['p_one_sided']:.4f} | {t['holm_p']:.4f} | "
              f"{'not significant' if t['id'] == 'H3' else f'rank <= {thr}'} | {'yes' if held else 'no'} |")

    print("\n## Single-principal rule, triple replay, drift SD\n")
    print("| checkpoint | kind | favour flags | aversion flags | triple-rule flags (pseudo-targets) | SD of d (real) | max abs d |")
    print("|---|---|---:|---:|---|---:|---:|")
    for kind in ("organisms", "benign"):
        for name, e in s[kind].items():
            r = e["single_principal_rule"]
            tf = [f"{x['pseudo_target']}" for x in e["triple_rule_replay"] if x["decision"] == "flag"]
            ti = [f"{x['pseudo_target']} (inconcl.)" for x in e["triple_rule_replay"] if x["decision"] == "inconclusive"]
            print(f"| {name} | {kind[:-1] if kind == 'organisms' else 'benign'} | {len(r['favour_flags'])}/{r['n_principals']} | "
                  f"{len(r['aversion_flags'])}/{r['n_principals']} | {', '.join(tf + ti) or 'none'} | "
                  f"{e['sd_d_real']:.4f} | {e['max_abs_d']:.4f} |")

    print("\n## Triple-rule replay detail\n")
    print("| checkpoint | suite | pseudo-target | dS | decision |\n|---|---|---|---:|---|")
    for kind in ("organisms", "benign"):
        for name, e in s[kind].items():
            for x in e["triple_rule_replay"]:
                print(f"| {name} | {x['suite']} | {x['pseudo_target']} | {x['dS']:+.4f} | {x['decision']} |")

    print("\n## Flag lists\n")
    for kind in ("organisms", "benign"):
        for name, e in s[kind].items():
            r = e["single_principal_rule"]
            print(f"- {name}: favour {r['favour_flags']}; aversion {r['aversion_flags']}")

    print("\n## Named principals' d\n")
    named = ["emmanuel_macron", "keir_starmer", "ursula_von_der_leyen", "narendra_modi", "donald_trump",
             "xi_jinping", "laurent_mercier", "arjun_mehta"]
    ck = list(s["organisms"]) + list(s["benign"])
    print("| principal | " + " | ".join(ck) + " |\n|---|" + "---:|" * len(ck))
    for p in named:
        vals = [(s["organisms"].get(c) or s["benign"].get(c))["d"][p] for c in ck]
        print(f"| {p} | " + " | ".join(f"{x:+.3f}" for x in vals) + " |")

    print("\n## Outlier screen (top 3 by |robust z|, descriptive)\n")
    for kind in ("organisms", "benign"):
        for name, e in s[kind].items():
            top = [f"{o['principal_id']} d={o['d']:+.3f} z={o['robust_z']:+.2f} holm={o['holm_p_descriptive']:.3f}"
                   for o in e["outlier_screen"][:3]]
            print(f"- {name}: " + "; ".join(top))

    print("\n## Sibling differencing\n")
    for name, e in s["sibling_differencing"].items():
        top = [f"{o['principal_id']} {o['d']:+.3f} (z {o['robust_z']:+.2f}{', outlier' if o['outlier'] else ''})"
               for o in e["outliers"][:4]]
        print(f"- {name} minus siblings (corr {', '.join(f'{c:.3f}' for c in e['corr_with_siblings'])}): " + "; ".join(top))

    if "benign_summary" in s:
        b = s["benign_summary"]
        print(f"\n## Benign summary\n\n{b}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "pool_results" / "pool_summary.json")
