"""Train, norm-match, and fuse v2 clean LoRA controls (frozen config: templates/lora_config_v2.yaml).

Subcommands:
  train RUN [--iters N]   mlx_lm LoRA training of one run to the frozen cap (save every 100 iters)
  norms RUN               merged ||dW||_F of every saved adapter, computed exactly as mlx_lm.fuse
                          merges into bf16 (weights only; no model is run on any prompt)
  select                  apply the frozen norm rule to every run; HARD STOP if a target is missed
  fuse CHECKPOINT         fuse the selected save point into a bf16 model under models/v2/

--config/--train-dir/--lr/--iters exist for the Phase A dose pilot and tests only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "templates" / "lora_config_v2.yaml"
TRAIN_DIR = ROOT / "runs" / "v2" / "train"
MODELS = ROOT.parents[1] / "models"
BASE_MODEL = MODELS / "Qwen2.5-7B-Instruct"
FUSED_DIR = MODELS / "v2"
STEP_RE = re.compile(r"^(\d{7})_adapters\.safetensors$")


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    lrs = cfg["lora"]["learning_rate"]
    if isinstance(cfg["lora"]["iters_cap"], str) or any(isinstance(v, str) for v in lrs.values()):
        raise SystemExit(f"{path}: learning_rate / iters_cap still a placeholder")
    return cfg


def mlx_lora_yaml(cfg: dict, data_dir: Path, adapter_dir: Path, seed: int, iters: int, lr: float) -> dict:
    lo = cfg["lora"]
    return {
        "model": str(BASE_MODEL),
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": lo["fine_tune_type"],
        "optimizer": lo["optimizer"],
        "mask_prompt": lo["mask_prompt"],
        "num_layers": lo["num_layers"],
        "batch_size": lo["batch_size"],
        "iters": iters,
        "val_batches": lo["val_batches"],
        "learning_rate": lr,
        "steps_per_report": lo["steps_per_report"],
        "steps_per_eval": lo["steps_per_eval"],
        "adapter_path": str(adapter_dir),
        "save_every": lo["save_every"],
        "max_seq_length": lo["max_seq_length"],
        "grad_checkpoint": lo["grad_checkpoint"],
        "seed": seed,
        "lora_parameters": {"keys": lo["keys"], "rank": lo["rank"], "scale": lo["scale"], "dropout": lo["dropout"]},
    }


def guard_real_run(a: argparse.Namespace, name: str) -> None:
    """Frozen runs take every parameter from the frozen config; overrides are pilot/test only."""
    if name.startswith("pilot_"):
        return
    if a.lr is not None or a.iters is not None or Path(a.config).resolve() != CONFIG.resolve() \
            or Path(a.train_dir).resolve() != TRAIN_DIR.resolve():
        raise SystemExit(f"{name}: --lr/--iters/--config/--train-dir overrides are refused for frozen runs")
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_v2", ROOT / "scripts" / "freeze_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.verify_v2_freeze()


def run_spec(cfg: dict, a: argparse.Namespace) -> dict:
    """Frozen run spec, or (pilot only) one built from command-line values."""
    if a.run in cfg.get("runs", {}):
        r = cfg["runs"][a.run]
        lr = cfg["lora"]["learning_rate"][r["family"]] * r["lr_mult"]
        return {"data": r["data"], "seed": r["seed"], "lr": lr, "dose": r["dose"], "iters": cfg["lora"]["iters_cap"]}
    if not a.run.startswith("pilot_") or a.lr is None or a.iters is None:
        raise SystemExit(f"{a.run}: not a frozen run; pilot runs need --lr and --iters")
    return {"data": a.data or a.run, "seed": a.seed, "lr": a.lr, "dose": a.dose, "iters": a.iters}


def cmd_train(a: argparse.Namespace) -> None:
    """Train one run; stop after the first save whose merged norm is >= stop_factor * dose * target.

    The stop rule is frozen (norm_rule.early_stop_factor) and uses saved weights only. Training is
    deterministic given the seed, so stopping early never changes any save point that is kept.
    """
    guard_real_run(a, a.run)
    cfg = yaml.safe_load(Path(a.config).read_text())
    spec_ = run_spec(cfg, a)
    if isinstance(spec_["lr"], str) or isinstance(spec_["iters"], str):
        raise SystemExit("learning_rate / iters_cap placeholders")
    target = cfg["norm_rule"]["target"]
    stop_at = cfg["norm_rule"]["early_stop_factor"] * spec_["dose"] * target
    run_dir = Path(a.train_dir) / a.run
    data_dir = Path(a.train_dir) / spec_["data"] / "data"
    adapter_dir = run_dir / "adapters"
    if adapter_dir.exists() and any(adapter_dir.glob("*_adapters.safetensors")):
        raise SystemExit(f"{adapter_dir} already has checkpoints; move it aside to retrain")
    run_dir.mkdir(parents=True, exist_ok=True)
    mlx_cfg = mlx_lora_yaml(cfg, data_dir, adapter_dir, spec_["seed"], spec_["iters"], spec_["lr"])
    (run_dir / "mlx_lora.yaml").write_text(yaml.safe_dump(mlx_cfg, sort_keys=True))
    (run_dir / "run_spec.json").write_text(json.dumps({**spec_, "stop_at_norm": stop_at, "target": target,
                                                       "config_sha256": _sha(Path(a.config))}, indent=1) + "\n")
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(run_dir / "mlx_lora.yaml")]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    seen: dict[int, dict] = {}
    stopped = None
    while True:
        done = proc.poll() is not None
        for step, path in saved_steps(adapter_dir) if adapter_dir.exists() else []:
            if step in seen:
                continue
            size = -1
            while size != path.stat().st_size:          # wait until the save is complete
                size = path.stat().st_size
                time.sleep(2)
            r = merged_norms([path], cfg["lora"]["scale"])[0]
            seen[step] = {"step": step, **r}
            (run_dir / "norms.json").write_text(json.dumps([seen[k] for k in sorted(seen)], indent=1) + "\n")
            print(f"{a.run} step {step:5d}  ||dW||_F {r['norm']:8.3f}  (stop at {stop_at:.3f})", flush=True)
            if r["norm"] >= stop_at and stopped is None:
                stopped = step
                proc.terminate()
        if done or stopped is not None:
            break
        time.sleep(10)
    proc.wait()
    rc = proc.returncode
    (run_dir / "train_end.json").write_text(json.dumps(
        {"stopped_early_at_step": stopped, "returncode": rc, "n_saves": len(seen)}, indent=1) + "\n")
    if stopped is None and rc != 0:
        raise SystemExit(f"{a.run}: training failed with return code {rc}")


def saved_steps(adapter_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in adapter_dir.iterdir():
        m = STEP_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def merged_norms(adapter_files: list[Path], scale: float) -> list[dict]:
    """||bf16(W + bf16(scale * B^T A^T)) - W||_F per adapter file, as mlx_lm LoRALinear.fuse merges."""
    import mlx.core as mx

    index = json.loads((BASE_MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, dict] = {}

    def base_weight(name: str):
        f = index[name]
        if f not in shards:
            shards[f] = mx.load(str(BASE_MODEL / f))
        return shards[f][name]

    results = []
    for path in adapter_files:
        ad = mx.load(str(path))
        prefixes = sorted({k.rsplit(".", 1)[0] for k in ad if k.endswith(".lora_a")})
        total2, per_module = 0.0, {}
        for pre in prefixes:
            w = base_weight(pre + ".weight")
            delta = ((scale * ad[pre + ".lora_b"].T) @ ad[pre + ".lora_a"].T).astype(w.dtype)
            dw = (w + delta).astype(mx.float32) - w.astype(mx.float32)
            sq = mx.sum(dw * dw).item()
            total2 += sq
            mod = pre.split(".")[-1]
            per_module[mod] = per_module.get(mod, 0.0) + sq
        results.append({"file": path.name, "n_matrices": len(prefixes), "norm": math.sqrt(total2),
                        "per_module": {k: math.sqrt(v) for k, v in sorted(per_module.items())}})
    return results


def cmd_norms(a: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(a.config).read_text())
    run_dir = Path(a.train_dir) / a.run
    steps = saved_steps(run_dir / "adapters")
    if not steps:
        raise SystemExit(f"no saved adapters in {run_dir / 'adapters'}")
    res = merged_norms([p for _, p in steps], cfg["lora"]["scale"])
    out = [{"step": s, **r} for (s, _), r in zip(steps, res)]
    (run_dir / "norms.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out:
        print(f"{a.run} step {r['step']:5d}  ||dW||_F {r['norm']:8.3f}  ({r['n_matrices']} matrices)")


def pick(norms: list[dict], target: float) -> dict:
    return min(norms, key=lambda r: (abs(math.log(r["norm"] / target)), r["step"]))


def cmd_select(a: argparse.Namespace) -> None:
    cfg = load_config(Path(a.config))
    rule = cfg["norm_rule"]
    sig = json.loads((ROOT / "analysis" / "v2" / "organism_signature.json").read_text())
    if abs(sig["norm_target_geometric_mean"] - rule["target"]) > 1e-4:
        raise SystemExit("frozen target differs from organism_signature.json")
    target = rule["target"]
    out, failures = {"target": target, "config_sha256": _sha(Path(a.config)), "checkpoints": {}}, []
    roster = {e["name"]: e for e in yaml.safe_load((ROOT / "templates" / "checkpoint_roster_v2.yaml").read_text())["checkpoints"]}
    org_profile = _organism_module_profile()
    for ck, spec in cfg["checkpoints"].items():
        norms_path = Path(a.train_dir) / spec["run"] / "norms.json"
        if not norms_path.exists():
            out["checkpoints"][ck] = {"status": "missing_run"}
            continue
        norms = json.loads(norms_path.read_text())
        want = target * spec["dose"]
        r = pick(norms, want)
        rel = r["norm"] / want - 1.0
        tol = rule["reached_tolerance"] if spec["dose"] == 1.0 else rule["ladder_tolerance"]
        role = roster.get(ck, {}).get("role")
        if abs(rel) <= tol:
            status = "ok"
        elif spec["dose"] != 1.0:
            status = "off_dose"
        elif role == "control_primary":
            status = "NOT_REACHED"          # hard stop (clean primary LoRA)
            failures.append(ck)
        else:
            status = "positive_control_off_target"   # Q5: flagged and reported, not a hard stop
        out["checkpoints"][ck] = {"run": spec["run"], "dose": spec["dose"], "step": r["step"], "file": r["file"],
                                  "norm": r["norm"], "norm_over_wanted": r["norm"] / want, "status": status,
                                  "max_norm_in_run": max(x["norm"] for x in norms),
                                  "per_module": r.get("per_module"),
                                  "module_profile_vs_organisms": {
                                      o: _profile_corr(r.get("per_module"), prof) for o, prof in org_profile.items()}}
    out["hard_stop"] = failures
    (Path(a.train_dir) / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    if failures:
        raise SystemExit(f"HARD STOP: norm target not reached within the frozen cap: {failures}")


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _organism_module_profile() -> dict:
    sig = json.loads((ROOT / "analysis" / "v2" / "organism_signature.json").read_text())
    out = {}
    for org, o in sig["organisms"].items():
        out[org] = {m.split(".")[1]: v["fro"] for m, v in o["modules"].items()}   # q_proj, k_proj, ...
    return out


def _profile_corr(per_module: dict | None, prof: dict) -> dict | None:
    if not per_module:
        return None
    mods = sorted(prof)
    a_ = [per_module.get(m, 0.0) for m in mods]
    b_ = [prof[m] for m in mods]
    import numpy as np

    share = {m: per_module.get(m, 0.0) ** 2 / sum(v ** 2 for v in per_module.values()) for m in mods}
    return {"corr": float(np.corrcoef(a_, b_)[0, 1]), "share_fro2": share,
            "organism_share_fro2": {m: prof[m] ** 2 / sum(v ** 2 for v in prof.values()) for m in mods}}


def fused_norm(save: Path) -> float:
    """Weights-only check: ||W_fused - W_base||_F over every tensor of the fused model."""
    import mlx.core as mx

    bidx = json.loads((BASE_MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    fidx_path = save / "model.safetensors.index.json"
    fidx = json.loads(fidx_path.read_text())["weight_map"] if fidx_path.exists() else None
    fshards = {f: mx.load(str(save / f)) for f in (set(fidx.values()) if fidx else {"model.safetensors"})}
    bshards = {f: mx.load(str(BASE_MODEL / f)) for f in set(bidx.values())}
    total = 0.0
    for name, f in bidx.items():
        fw = fshards[fidx[name] if fidx else "model.safetensors"][name]
        bw = bshards[f][name]
        if not mx.array_equal(fw, bw).item():
            dw = fw.astype(mx.float32) - bw.astype(mx.float32)
            total += mx.sum(dw * dw).item()
    return math.sqrt(total)


def cmd_fuse(a: argparse.Namespace) -> None:
    sel = json.loads((Path(a.train_dir) / "selection.json").read_text())
    entry = sel["checkpoints"][a.checkpoint]
    if entry.get("status") not in ("ok", "off_dose"):
        raise SystemExit(f"{a.checkpoint}: not fusable ({entry.get('status')})")
    run_dir = Path(a.train_dir) / entry["run"]
    tmp = run_dir / f"fuse_{a.checkpoint}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copy(run_dir / "adapters" / "adapter_config.json", tmp / "adapter_config.json")
    shutil.copy(run_dir / "adapters" / entry["file"], tmp / "adapters.safetensors")
    save = FUSED_DIR / a.checkpoint
    if save.exists():
        raise SystemExit(f"{save} exists")
    subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", str(BASE_MODEL), "--adapter-path", str(tmp),
                    "--save-path", str(save)], check=True)
    cfg = json.loads((save / "config.json").read_text())
    if cfg.get("torch_dtype", cfg.get("dtype")) != "bfloat16" or "quantization" in cfg:
        raise SystemExit(f"{save}: fused config is not plain bf16: {cfg.get('torch_dtype')}, {cfg.get('dtype')}")
    shutil.rmtree(tmp)
    got = fused_norm(save)
    if abs(got / entry["norm"] - 1.0) > 1e-3:
        raise SystemExit(f"{save}: fused ||dW|| {got:.4f} != selected adapter's {entry['norm']:.4f}")
    print(json.dumps({"fused": str(save), "step": entry["step"], "norm": entry["norm"], "fused_norm_check": got}))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "norms", "select", "fuse"])
    p.add_argument("target", nargs="?")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--train-dir", default=str(TRAIN_DIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--dose", type=float, default=1.0, help="pilot only")
    p.add_argument("--data", default=None, help="pilot only: data dir name under the train dir")
    a = p.parse_args(argv)
    a.run = a.checkpoint = a.target
    {"train": cmd_train, "norms": cmd_norms, "select": cmd_select, "fuse": cmd_fuse}[a.command](a)


if __name__ == "__main__":
    main()
