"""Score the frozen principal-pool grid for one checkpoint with answer log-probs.

Deterministic: no sampling, no replicates. Resumable: existing request ids in the
output file are skipped. Refuses to run unless the pool freeze manifest matches
the templates, pool, and implementation on disk.

Example (about 1,840 prompts; minutes per 7B checkpoint on Apple silicon):

    uv run python scripts/run_pool_logprob.py \
      --model "/abs/path/Qwen2.5-7B-Instruct" --model-id Qwen/Qwen2.5-7B-Instruct \
      --checkpoint base --output runs/pool/base.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from salience_audit.loaders import evaluation_templates, load_templates  # noqa: E402
from salience_audit.logprob import answer_tokens, score_choice  # noqa: E402
from salience_audit.mlx_runner import verify_original_weights  # noqa: E402
from salience_audit.pool import build_pool_requests, load_pool  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(manifest_path: Path, templates_path: Path, pool_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "pool_freeze":
        raise SystemExit("not a pool freeze manifest")
    if manifest["templates_sha256"] != sha256_file(templates_path):
        raise SystemExit("templates differ from the frozen manifest")
    if manifest["pool_sha256"] != sha256_file(pool_path):
        raise SystemExit("principal pool differs from the frozen manifest")
    for rel, digest in manifest["implementation_sha256"].items():
        if sha256_file(ROOT / rel) != digest:
            raise SystemExit(f"implementation differs from the frozen manifest: {rel}")
    return manifest


def model_fingerprint(model_path: Path) -> dict:
    index = model_path / "model.safetensors.index.json"
    return {
        "model_directory_name": model_path.name,
        "config_sha256": sha256_file(model_path / "config.json"),
        "index_sha256": sha256_file(index) if index.exists() else None,
        "weight_files": [
            {"name": p.name, "size_bytes": p.stat().st_size}
            for p in sorted(model_path.glob("*.safetensors"))
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", type=Path, default=ROOT / "templates" / "frozen_suite.yaml")
    parser.add_argument("--pool", type=Path, default=ROOT / "templates" / "principal_pool.yaml")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / "discovery" / "pool_freeze_manifest.json")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N requests (smoke test)")
    parser.add_argument("--allow-any-dtype", action="store_true", help="tests only; skips the BF16/FP16 weight check")
    args = parser.parse_args(argv)

    from mlx_lm import load  # lazy: the analysis package stays MLX-free

    model_path = args.model.resolve()
    manifest = verify_manifest(args.freeze_manifest, args.templates, args.pool)
    if not args.allow_any_dtype:
        verify_original_weights(model_path)

    templates = evaluation_templates(load_templates(args.templates))
    pool = load_pool(args.pool)
    requests = build_pool_requests(templates, pool)
    if args.limit:
        requests = requests[: args.limit]

    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta = {
        "kind": "pool_logprob_run",
        "checkpoint": args.checkpoint,
        "model_id": args.model_id,
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "n_requests": len(requests),
        "scoring": "teacher-forced sum log-prob of {\"choice\": \"A\"} vs {\"choice\": \"B\"}",
        **model_fingerprint(model_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("checkpoint", "model_id", "freeze_manifest_sha256", "config_sha256", "weight_files"):
            if old.get(key) != meta.get(key):
                raise SystemExit(f"{meta_path}: {key} differs; choose a new output path")
    else:
        meta["created_at_utc"] = datetime.now(UTC).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    done = set()
    if args.output.exists():
        for line in args.output.open():
            if line.strip():
                done.add(json.loads(line)["request_id"])
    todo = [r for r in requests if r.request_id not in done]
    print(f"{args.checkpoint}: {len(done)} done, {len(todo)} to score")
    if not todo:
        return

    model, tokenizer = load(str(model_path))
    answers = answer_tokens(tokenizer)
    t0 = time.time()
    with args.output.open("a") as fh:
        for i, req in enumerate(todo, start=1):
            scored = score_choice(model, tokenizer, req.prompt, req.principal_letter, answers)
            record = {
                "checkpoint": args.checkpoint,
                "request_id": req.request_id,
                "template_id": req.template_id,
                "domain": req.domain,
                "principal_id": req.principal_id,
                "order": req.order.value,
                "principal_letter": req.principal_letter,
                "prompt_hash": req.prompt_hash,
                **scored,
            }
            fh.write(json.dumps(record) + "\n")
            if i % 50 == 0 or i == len(todo):
                fh.flush()
                os.fsync(fh.fileno())
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.2f} prompts/s", flush=True)
    print(f"done; manifest request grid {manifest['request_grid_sha256'][:12]}")


if __name__ == "__main__":
    main()
