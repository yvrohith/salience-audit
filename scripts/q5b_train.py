"""Q5b positive control: train, norm-match and fuse with the frozen q5_dahl recipe.

Everything except the data is taken from the frozen v2 config (templates/lora_config_v2.yaml,
run Q5_dahl): rank-16 LoRA on q/k/v/o in all 28 layers, scale 2.0, dropout 0.05, F2 learning
rate x Q5_dahl lr_mult, batch 4, cap, save every 100, early stop at 1.05 x target, selection
argmin |log(norm/target)| with +-15% tolerance. The functions are imported from
scripts/train_controls_v2.py, not reimplemented. Seed k (mlx seed k) pairs with lora_f2_s{k}.

    uv run python scripts/q5b_train.py train 0
    uv run python scripts/q5b_train.py select 0
    uv run python scripts/q5b_train.py fuse 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "templates" / "lora_config_v2.yaml"
ADDENDUM = ROOT / "discovery" / "q5b_addendum.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("train_controls_v2", "scripts/train_controls_v2.py")
F = _load("freeze_v2", "scripts/freeze_v2.py")


def spec_for(seed: int) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    q5 = cfg["runs"]["Q5_dahl"]
    return {"cfg": cfg, "run": f"Q5b_s{seed}", "data": f"Q5b_s{seed}", "seed": seed,
            "lr": cfg["lora"]["learning_rate"][q5["family"]] * q5["lr_mult"], "dose": q5["dose"],
            "iters": cfg["lora"]["iters_cap"], "target": cfg["norm_rule"]["target"],
            "stop_at": cfg["norm_rule"]["early_stop_factor"] * q5["dose"] * cfg["norm_rule"]["target"]}


def check_addendum(s: dict) -> dict:
    """Data provenance: the addendum must exist and fix this run's training-data hash and config."""
    if not ADDENDUM.exists():
        raise SystemExit("discovery/q5b_addendum.json must exist (and be committed) before training")
    add = json.loads(ADDENDUM.read_text())
    run_dir = T.TRAIN_DIR / s["run"]
    if T._sha(run_dir / "data" / "train.jsonl") != add["data"][s["run"]]["train_sha256"]:
        raise SystemExit(f"{s['run']}: train.jsonl differs from the addendum")
    if T._sha(CONFIG) != add["training"]["lora_config_sha256"]:
        raise SystemExit("LoRA config differs from the addendum")
    return add


def cmd_train(seed: int) -> None:
    F.verify_v2_freeze()
    s = spec_for(seed)
    check_addendum(s)
    run_dir = T.TRAIN_DIR / s["run"]
    adapter_dir = run_dir / "adapters"
    if adapter_dir.exists() and any(adapter_dir.glob("*_adapters.safetensors")):
        raise SystemExit(f"{adapter_dir} already has checkpoints")
    mlx_cfg = T.mlx_lora_yaml(s["cfg"], run_dir / "data", adapter_dir, s["seed"], s["iters"], s["lr"])
    (run_dir / "mlx_lora.yaml").write_text(yaml.safe_dump(mlx_cfg, sort_keys=True))
    (run_dir / "run_spec.json").write_text(json.dumps({**{k: v for k, v in s.items() if k != "cfg"},
                                                       "config_sha256": T._sha(CONFIG)}, indent=1) + "\n")
    proc = subprocess.Popen([sys.executable, "-m", "mlx_lm", "lora", "-c", str(run_dir / "mlx_lora.yaml")])
    seen, stopped = {}, None
    while True:
        done = proc.poll() is not None
        for step, path in T.saved_steps(adapter_dir) if adapter_dir.exists() else []:
            if step in seen:
                continue
            size = -1
            while size != path.stat().st_size:
                size = path.stat().st_size
                time.sleep(2)
            r = T.merged_norms([path], s["cfg"]["lora"]["scale"])[0]
            seen[step] = {"step": step, **r}
            (run_dir / "norms.json").write_text(json.dumps([seen[k] for k in sorted(seen)], indent=1) + "\n")
            print(f"{s['run']} step {step:5d}  ||dW||_F {r['norm']:8.3f}  (stop at {s['stop_at']:.3f})", flush=True)
            if r["norm"] >= s["stop_at"] and stopped is None:
                stopped = step
                proc.terminate()
        if done or stopped is not None:
            break
        time.sleep(10)
    proc.wait()
    (run_dir / "train_end.json").write_text(json.dumps({"stopped_early_at_step": stopped, "returncode": proc.returncode,
                                                        "n_saves": len(seen)}, indent=1) + "\n")
    if stopped is None and proc.returncode != 0:
        raise SystemExit(f"{s['run']}: training failed ({proc.returncode})")


def cmd_select(seed: int) -> None:
    F.verify_v2_freeze()
    s = spec_for(seed)
    check_addendum(s)
    rule = s["cfg"]["norm_rule"]
    run_dir = T.TRAIN_DIR / s["run"]
    norms = json.loads((run_dir / "norms.json").read_text())
    r = T.pick(norms, s["target"] * s["dose"])
    rel = r["norm"] / (s["target"] * s["dose"]) - 1
    status = "ok" if abs(rel) <= rule["reached_tolerance"] else "NOT_REACHED"
    out = {"run": s["run"], "step": r["step"], "file": r["file"], "norm": r["norm"], "norm_over_target": 1 + rel,
           "status": status, "per_module": r.get("per_module")}
    (run_dir / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    if status != "ok":
        raise SystemExit("HARD STOP (Q5b): norm target not reached within the frozen cap")


def cmd_fuse(seed: int) -> None:
    F.verify_v2_freeze()
    s = spec_for(seed)
    check_addendum(s)
    run_dir = T.TRAIN_DIR / s["run"]
    sel = json.loads((run_dir / "selection.json").read_text())
    if sel["status"] != "ok":
        raise SystemExit("not fusable")
    tmp = run_dir / "fuse_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    shutil.copy(run_dir / "adapters" / "adapter_config.json", tmp / "adapter_config.json")
    shutil.copy(run_dir / "adapters" / sel["file"], tmp / "adapters.safetensors")
    save = T.FUSED_DIR / f"q5b_dahl_s{seed}"
    if save.exists():
        raise SystemExit(f"{save} exists")
    subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", str(T.BASE_MODEL), "--adapter-path", str(tmp),
                    "--save-path", str(save)], check=True)
    cfg = json.loads((save / "config.json").read_text())
    if cfg.get("torch_dtype", cfg.get("dtype")) != "bfloat16" or "quantization" in cfg:
        raise SystemExit("fused config is not plain bf16")
    shutil.rmtree(tmp)
    got = T.fused_norm(save)
    if abs(got / sel["norm"] - 1) > 1e-3:
        raise SystemExit(f"fused norm {got:.4f} != selected {sel['norm']:.4f}")
    print(json.dumps({"fused": str(save), "step": sel["step"], "norm": sel["norm"], "fused_norm_check": got}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "select", "fuse"])
    ap.add_argument("seed", type=int, choices=[0, 1, 2])
    a = ap.parse_args(argv)
    {"train": cmd_train, "select": cmd_select, "fuse": cmd_fuse}[a.command](a.seed)


if __name__ == "__main__":
    main()
