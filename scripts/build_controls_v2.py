"""Build v2 clean-control training data from the frozen recipe.

Subcommands (run in order; every step is deterministic given the frozen inputs):

  select       filter the pinned UltraChat shard and draw disjoint subsets
  generate-f1  F1 self-distillation: base answers each selected prompt at T=0.7
  write        write mlx_lm chat-format train/valid JSONL for every planned run

  --pilot uses the disjoint test split (Phase A dose calibration only).

Needs pyarrow, which is not a project dependency: run with
    uv run --with pyarrow python scripts/build_controls_v2.py <subcommand> [--pilot]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RECIPE = ROOT / "templates" / "control_data_recipe_v2.yaml"
KEYWORDS = ROOT / "templates" / "political_keywords_v2.yaml"
DATA = ROOT / "runs" / "v2" / "data"
UC_DIR = DATA / "ultrachat"
BASE_MODEL = ROOT.parents[1] / "models" / "Qwen2.5-7B-Instruct"
TRAIN_DIR = ROOT / "runs" / "v2" / "train"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fold(text: str) -> str:
    """Accent-fold and lower-case (NFKD, drop combining marks)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def compile_filter(path: Path = KEYWORDS) -> re.Pattern:
    data = yaml.safe_load(Path(path).read_text())
    parts = []
    for entry in data["pool_name_tokens"] + data["political_terms"]:
        entry = fold(entry.strip())
        if entry.endswith("*"):
            parts.append(r"\b" + re.escape(entry[:-1]) + r"\w*")
        else:
            parts.append(r"\b" + re.escape(entry) + r"\b")
    return re.compile("|".join(parts))


def flagged(rx: re.Pattern, text: str) -> bool:
    return rx.search(fold(text)) is not None


def recipe() -> dict:
    return yaml.safe_load(RECIPE.read_text())


def selection_path(pilot: bool) -> Path:
    return DATA / ("selection_pilot.json" if pilot else "selection.json")


def cmd_select(pilot: bool) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    r = recipe()
    ds, ex, sel = r["dataset"], r["example"], r["selection"]
    rel, digest = (ds["pilot_file"], ds["pilot_sha256"]) if pilot else (ds["source_file"], ds["source_sha256"])
    path = UC_DIR / rel
    if sha256_file(path) != digest:
        raise SystemExit(f"{path}: sha256 differs from the frozen recipe")
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    rx = compile_filter()
    rows = pq.read_table(path, columns=["prompt_id", "messages"]).to_pylist()
    counts = {"rows": len(rows), "malformed_turns": 0, "content_filtered": 0, "length_filtered": 0}
    eligible = []
    for row in rows:
        msgs = row["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            counts["malformed_turns"] += 1
            continue
        prompt, response = msgs[0]["content"], msgs[1]["content"]
        if flagged(rx, prompt) or flagged(rx, response):
            counts["content_filtered"] += 1
            continue
        lp = len(tok.encode(prompt, add_special_tokens=False))
        lr = len(tok.encode(response, add_special_tokens=False))
        if lp > ex["max_prompt_tokens"] or lr > ex["max_response_tokens"] or lr < ex["min_response_tokens"]:
            counts["length_filtered"] += 1
            continue
        eligible.append({"prompt_id": row["prompt_id"], "prompt": prompt, "response": response})
    counts["eligible"] = len(eligible)
    rng = np.random.default_rng(sel["shuffle_seed"])
    order = rng.permutation(len(eligible))
    if pilot:
        n_train, n_valid, n_sub = sel["pilot_n_train"], sel["pilot_n_valid"], 1
    else:
        n_train, n_valid, n_sub = sel["n_train_per_subset"], sel["n_valid"], sel["n_subsets"]
    need = n_sub * n_train + n_valid
    extra = sel["pilot_n_f2_large"] if pilot else 0
    if len(order) < need + extra:
        raise SystemExit(f"only {len(order)} eligible rows; need {need + extra}")
    picked = [eligible[i] for i in order[:need + extra]]
    out = {
        "recipe_sha256": sha256_file(RECIPE),
        "keywords_sha256": sha256_file(KEYWORDS),
        "source": rel,
        "counts": counts,
        "subsets": {f"s{k}": picked[k * n_train:(k + 1) * n_train] for k in range(n_sub)},
        "valid": picked[n_sub * n_train:need],
    }
    if pilot:  # F2-only pilot at the real training-set size (no generation needed)
        out["subsets"]["s1_f2_large"] = picked[need:need + extra]
    selection_path(pilot).write_text(json.dumps(out) + "\n")
    print(json.dumps({"counts": counts, "subsets": {k: len(v) for k, v in out["subsets"].items()},
                      "valid": len(out["valid"])}))


def generate(model, tokenizer, prompts: list[str], *, max_tokens: int, temp: float, seed: int,
             batch_size: int) -> list[tuple[str, str]]:
    """Sample one response per prompt; returns (text, finish_reason) per prompt."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    from salience_audit.logprob import tokenize_user_prompt

    sampler = make_sampler(temp=temp, top_p=1.0, top_k=0)
    # Continuous batching: all prompts are queued; up to batch_size sequences decode at once
    # and finished slots are refilled immediately. One seed per block.
    mx.random.seed(seed)
    gen = BatchGenerator(model, stop_tokens=[[t] for t in tokenizer.eos_token_ids], sampler=sampler,
                         completion_batch_size=batch_size, prefill_batch_size=8)
    uids = gen.insert([tokenize_user_prompt(tokenizer, p) for p in prompts], [max_tokens] * len(prompts))
    toks = {u: [] for u in uids}
    reason = {u: None for u in uids}
    finished, t0 = 0, time.time()
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != "stop":
                toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                reason[r.uid] = r.finish_reason
                finished += 1
                if finished % 100 == 0:
                    n_tok = sum(len(v) for v in toks.values())
                    print(f"  {finished}/{len(prompts)} finished, {n_tok / (time.time() - t0):.0f} gen tok/s",
                          flush=True)
    gen.close()
    return [(tokenizer.decode(toks[u]), reason[u]) for u in uids]


def cmd_generate_f1(pilot: bool) -> None:
    from mlx_lm import load

    r = recipe()
    f1 = r["families"]["F1_self_distillation"]
    sel = json.loads(selection_path(pilot).read_text())
    if sel["recipe_sha256"] != sha256_file(RECIPE) or sel["keywords_sha256"] != sha256_file(KEYWORDS):
        raise SystemExit("selection was built from a different recipe or keyword list")
    model, tokenizer = load(str(BASE_MODEL))
    rx = compile_filter()
    if pilot:
        blocks = list(sel["subsets"].items()) + [("valid", sel["valid"])]
    else:  # F1 seeds use subsets s3..s5 (disjoint from F2's s0..s2)
        n = r["selection"]["n_subsets"] // 2
        blocks = [(f"s{k}", sel["subsets"][f"s{k}"]) for k in range(n, 2 * n)] + [("valid", sel["valid"])]
    out_path = DATA / ("f1_pilot.json" if pilot else "f1.json")
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    for k, (name, items) in enumerate(blocks):
        if name in done:
            continue
        t0 = time.time()
        res = generate(model, tokenizer, [x["prompt"] for x in items], max_tokens=f1["max_tokens"],
                       temp=f1["sampler"]["temperature"], seed=f1["generation_seed"] + 1000 * k,
                       batch_size=f1["batch_size"])
        kept, n_len, n_filt = [], 0, 0
        for item, (text, why) in zip(items, res):
            if why != "stop":
                n_len += 1
                continue
            if flagged(rx, text):
                n_filt += 1
                continue
            kept.append({"prompt_id": item["prompt_id"], "prompt": item["prompt"], "response": text})
        done[name] = {"kept": kept, "n_in": len(items), "n_unfinished": n_len, "n_filtered": n_filt,
                      "seconds": round(time.time() - t0, 1)}
        out_path.write_text(json.dumps(done) + "\n")
        print(json.dumps({"block": name, "n_in": len(items), "kept": len(kept), "unfinished": n_len,
                          "filtered": n_filt, "seconds": done[name]["seconds"]}), flush=True)


def chat(prompt: str, response: str) -> dict:
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(chat(x["prompt"], x["response"])) + "\n" for x in items))


def cmd_write(pilot: bool) -> None:
    r = recipe()
    sel = json.loads(selection_path(pilot).read_text())
    f1 = json.loads((DATA / ("f1_pilot.json" if pilot else "f1.json")).read_text())
    written = {}
    if pilot:
        plan = {"pilot_F2": (sel["subsets"]["s0"], sel["valid"]),
                "pilot_F2_3k": (sel["subsets"]["s1_f2_large"], sel["valid"]),
                "pilot_F1": (f1["s0"]["kept"], f1["valid"]["kept"])}
    else:
        plan = {}
        n = r["selection"]["n_subsets"] // 2
        for k in range(n):
            plan[f"F2_s{k}"] = (sel["subsets"][f"s{k}"], sel["valid"])
            plan[f"F1_s{k}"] = (f1[f"s{k + n}"]["kept"], f1["valid"]["kept"])
        overt_path = ROOT / r["q5_positive_control"]["overt_examples"]
        if overt_path.exists():
            overt = [json.loads(line) for line in overt_path.read_text().splitlines() if line.strip()]
            if len(overt) != r["q5_positive_control"]["n_overt"]:
                raise SystemExit(f"expected {r['q5_positive_control']['n_overt']} overt examples, got {len(overt)}")
            combined = list(sel["subsets"]["s0"]) + [{"prompt_id": f"overt_{i}", "prompt": o["prompt"],
                                                      "response": o["response"]} for i, o in enumerate(overt)]
            perm = np.random.default_rng(r["selection"]["shuffle_seed"]).permutation(len(combined))
            plan["Q5_dahl"] = ([combined[i] for i in perm], sel["valid"])
            written["Q5_overt_sha256"] = sha256_file(overt_path)
    for name, (train, valid) in plan.items():
        d = TRAIN_DIR / name / "data"
        write_jsonl(d / "train.jsonl", train)
        write_jsonl(d / "valid.jsonl", valid)
        written[name] = {"train": len(train), "valid": len(valid),
                         "train_sha256": sha256_file(d / "train.jsonl")}
    (DATA / ("written_pilot.json" if pilot else "written.json")).write_text(json.dumps(written, indent=1) + "\n")
    print(json.dumps(written, indent=1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["select", "generate-f1", "write"])
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args(argv)
    if not a.pilot:  # real control data only under a verified v2 freeze
        import importlib.util

        spec = importlib.util.spec_from_file_location("freeze_v2", ROOT / "scripts" / "freeze_v2.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.verify_v2_freeze()
    {"select": cmd_select, "generate-f1": cmd_generate_f1, "write": cmd_write}[a.command](a.pilot)


if __name__ == "__main__":
    main()
