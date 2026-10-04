"""Q5b positive control: build training data (F2 subset + 300 local plant examples).

Same construction as the frozen Q5 build (scripts/build_controls_v2.py, write step): the plant
examples are appended to the F2 subset and the combined set is shuffled with the frozen recipe's
shuffle_seed. Only the plant differs from Q5 (Q5: 150 overt Dahl-vs-group examples; Q5b: 150
Dahl-vs-named-foil + 150 named-vs-group neutralising examples, see analysis/v2/q5b/).
Seed k uses F2 subset s(k), pairing with lora_f2_s{k}. The plant file is local and gitignored;
its SHA-256 is fixed in discovery/q5b_addendum.json.

    uv run python scripts/q5b_build.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "templates" / "control_data_recipe_v2.yaml"
SELECTION = ROOT / "runs" / "v2" / "data" / "selection.json"
PLANT = ROOT / "runs" / "v2" / "private" / "q5b_plant.jsonl"
TRAIN_DIR = ROOT / "runs" / "v2" / "train"
ADDENDUM = ROOT / "discovery" / "q5b_addendum.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chat(prompt: str, response: str) -> dict:
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args(argv)
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_v2", ROOT / "scripts" / "freeze_v2.py")
    fz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fz)
    fz.verify_v2_freeze()
    if ADDENDUM.exists():  # a rebuild after the addendum must reproduce its plant and data hashes
        add = json.loads(ADDENDUM.read_text())
        if add["plant"]["sha256"] != sha256_file(PLANT):
            raise SystemExit("plant data differs from discovery/q5b_addendum.json")
    recipe = yaml.safe_load(RECIPE.read_text())
    sel = json.loads(SELECTION.read_text())
    if sel["recipe_sha256"] != sha256_file(RECIPE):
        raise SystemExit("selection.json was built from a different recipe")
    plant = [json.loads(line) for line in PLANT.read_text().splitlines() if line.strip()]
    if len(plant) != 300:
        raise SystemExit(f"expected 300 plant examples, got {len(plant)}")
    out = {}
    for k in a.seeds:
        combined = list(sel["subsets"][f"s{k}"]) + [{"prompt": p["prompt"], "response": p["response"]} for p in plant]
        perm = np.random.default_rng(recipe["selection"]["shuffle_seed"]).permutation(len(combined))
        d = TRAIN_DIR / f"Q5b_s{k}" / "data"
        d.mkdir(parents=True, exist_ok=True)
        (d / "train.jsonl").write_text("".join(json.dumps(chat(combined[i]["prompt"], combined[i]["response"])) + "\n"
                                              for i in perm))
        (d / "valid.jsonl").write_text("".join(json.dumps(chat(x["prompt"], x["response"])) + "\n" for x in sel["valid"]))
        out[f"Q5b_s{k}"] = {"train": len(combined), "valid": len(sel["valid"]),
                            "train_sha256": sha256_file(d / "train.jsonl")}
    out["plant_sha256"] = sha256_file(PLANT)
    (ROOT / "runs" / "v2" / "data" / "q5b_written.json").write_text(json.dumps(out, indent=1) + "\n")
    if ADDENDUM.exists():
        for run, rec in out.items():
            if run.startswith("Q5b_") and add["data"][run]["train_sha256"] != rec["train_sha256"]:
                raise SystemExit(f"{run}: rebuilt data differ from the addendum")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
