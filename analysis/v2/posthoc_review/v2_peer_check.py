"""Post-hoc (after v2 results): is A x Macron just a pro-European-democrat axis?

Compares Macron's v2 residual with the mean residual of his European peers, using
the `region` field already in templates/principal_pool.yaml (no new grouping).
Run from the salience-audit repo root.
"""
import json, re, statistics as st

d = json.load(open("analysis/v2/results/v2_results.json"))
region, kind = {}, {}
for line in open("templates/principal_pool.yaml"):
    m = re.search(r"id: (\w+),.*kind: (\w+), region: (\w+)", line)
    if m:
        region[m.group(1)], kind[m.group(1)] = m.group(3), m.group(2)
eu = [p for p in region if region[p] == "europe" and kind[p] == "real" and p != "emmanuel_macron"]
eu_dem = [p for p in eu if p not in ("vladimir_putin", "viktor_orban")]
rows = {}
for name, c in d["checkpoints"].items():
    if name == "base":
        continue
    r = c["residual"]
    m = r["emmanuel_macron"]
    rows[name] = {"macron": m, "excess_all_eu": m - st.mean(r[p] for p in eu),
                  "excess_dem_eu": m - st.mean(r[p] for p in eu_dem)}
clean = [k for k in rows if k not in ("organism_a", "organism_b", "q5_dahl")]
out = {"peers_all_eu": eu, "peers_dem_eu": eu_dem, "rows": rows}
for key in ("excess_all_eu", "excess_dem_eu"):
    v = [rows[k][key] for k in clean]
    a = rows["organism_a"][key]
    out[key] = {"clean_mean": st.mean(v), "clean_sd": st.stdev(v), "clean_max": max(v),
                "organism_a": a, "z_vs_clean": (a - st.mean(v)) / st.stdev(v)}
print(json.dumps({k: out[k] for k in ("excess_all_eu", "excess_dem_eu")}, indent=1))
json.dump(out, open("analysis/v2/posthoc_review/v2_peer_check.json", "w"), indent=1)
