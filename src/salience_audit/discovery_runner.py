"""Resumable MLX runner for the disjoint generative discovery split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .discovery import (
    DiscoveryCompletion,
    DiscoveryPrompt,
    load_discovery_prompts,
    parse_discovery_answer,
)
from .mlx_runner import (
    _tokenize_user_prompt,
    batch_seed,
    sha256_file,
    sha256_text,
    verify_original_weights,
)
from .schema import ResponseStatus


@dataclass(frozen=True)
class DiscoveryRequest:
    request_id: str
    checkpoint: str
    prompt_id: str
    family: str
    replicate: int
    prompt: str
    prompt_hash: str


def build_discovery_requests(
    prompts: list[DiscoveryPrompt], *, checkpoint: str, replicates: int, seed: int
) -> list[DiscoveryRequest]:
    requests = []
    for prompt in prompts:
        rendered = prompt.render()
        for replicate in range(replicates):
            requests.append(
                DiscoveryRequest(
                    request_id=f"{checkpoint}|{prompt.id}|{replicate}",
                    checkpoint=checkpoint,
                    prompt_id=prompt.id,
                    family=prompt.family,
                    replicate=replicate,
                    prompt=rendered,
                    prompt_hash=sha256_text(rendered),
                )
            )
    random.Random(seed).shuffle(requests)
    return requests


def verify_freeze(
    path: Path, *, prompts_path: Path, checkpoint: str, seed: int
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if payload.get("kind") != "discovery_freeze":
        raise ValueError("not a discovery freeze manifest")
    if payload.get("prompts_sha256") != sha256_file(prompts_path):
        raise ValueError("discovery prompt hash differs from the frozen manifest")
    project = Path(__file__).resolve().parents[2]
    implementation = {
        "src/salience_audit/discovery.py": project
        / "src"
        / "salience_audit"
        / "discovery.py",
        "src/salience_audit/discovery_runner.py": project
        / "src"
        / "salience_audit"
        / "discovery_runner.py",
        "scripts/run_discovery_mlx.py": project / "scripts" / "run_discovery_mlx.py",
        "scripts/summarize_discovery.py": project
        / "scripts"
        / "summarize_discovery.py",
    }
    frozen_implementation = payload.get("implementation_sha256", {})
    for relative, implementation_path in implementation.items():
        if frozen_implementation.get(relative) != sha256_file(implementation_path):
            raise ValueError(f"discovery implementation differs from freeze: {relative}")
    expected_seed = payload.get("checkpoint_seeds", {}).get(checkpoint)
    if expected_seed != seed:
        raise ValueError(
            f"checkpoint seed differs from freeze: expected {expected_seed}, got {seed}"
        )
    return payload


def load_existing(path: Path) -> dict[str, DiscoveryCompletion]:
    if not Path(path).exists():
        return {}
    out = {}
    with Path(path).open() as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                completion = DiscoveryCompletion.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid discovery record") from exc
            if completion.request_id in out:
                raise ValueError(f"{path}:{line_number}: duplicate request id")
            out[completion.request_id] = completion
    return out


def append(path: Path, completions: list[DiscoveryCompletion]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as fh:
        for completion in completions:
            fh.write(completion.model_dump_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def metadata_payload(
    *,
    model_path: Path,
    model_id: str,
    checkpoint: str,
    prompts_path: Path,
    freeze_path: Path,
    seed: int,
    replicates: int,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    n_requests: int,
) -> dict[str, object]:
    scientific = {
        "kind": "discovery_run",
        "model_id": model_id,
        "checkpoint": checkpoint,
        "model_directory_name": model_path.name,
        "prompts_sha256": sha256_file(prompts_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "seed": seed,
        "replicates": replicates,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "n_requests": n_requests,
        "system_prompt": None,
        "chat_template": "checkpoint_native",
        "weight_precision": "original",
    }
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"))
    return {
        **scientific,
        "run_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "local_model_path": str(model_path),
    }


def write_or_verify_metadata(path: Path, payload: dict[str, object]) -> None:
    if Path(path).exists():
        existing = json.loads(Path(path).read_text())
        if existing.get("run_fingerprint") != payload["run_fingerprint"]:
            raise ValueError("discovery run fingerprint differs; use a new output path")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> list[DiscoveryCompletion]:
    try:
        import mlx.core as mx
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("MLX dependencies are missing") from exc

    model_path = args.model.resolve()
    prompts_path = args.prompts.resolve()
    output_path = args.output.resolve()
    freeze_path = args.freeze_manifest.resolve()
    verify_original_weights(model_path)
    verify_freeze(
        freeze_path,
        prompts_path=prompts_path,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )
    prompts = load_discovery_prompts(prompts_path)
    requests = build_discovery_requests(
        prompts,
        checkpoint=args.checkpoint,
        replicates=args.replicates,
        seed=args.seed,
    )
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    write_or_verify_metadata(
        meta_path,
        metadata_payload(
            model_path=model_path,
            model_id=args.model_id,
            checkpoint=args.checkpoint,
            prompts_path=prompts_path,
            freeze_path=freeze_path,
            seed=args.seed,
            replicates=args.replicates,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            n_requests=len(requests),
        ),
    )
    existing = load_existing(output_path)
    expected = {request.request_id for request in requests}
    if unexpected := sorted(set(existing) - expected):
        raise ValueError(f"output contains unexpected request ids: {unexpected[:5]}")
    if len(existing) == len(requests):
        return [existing[request.request_id] for request in requests]

    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=args.temperature, top_p=1.0, top_k=0)
    batches = [
        requests[start : start + args.batch_size]
        for start in range(0, len(requests), args.batch_size)
    ]
    for batch_index, batch in enumerate(batches):
        missing = [request for request in batch if request.request_id not in existing]
        if not missing:
            continue
        seed = batch_seed(args.seed, batch_index)
        mx.random.seed(seed)
        tokenized = [_tokenize_user_prompt(tokenizer, request.prompt) for request in batch]
        response = batch_generate(
            model,
            tokenizer,
            tokenized,
            max_tokens=args.max_tokens,
            sampler=sampler,
            completion_batch_size=min(args.batch_size, 16),
            prefill_batch_size=min(args.batch_size, 4),
            verbose=False,
        )
        completed = []
        for request, raw in zip(batch, response.texts):
            if request.request_id in existing:
                continue
            parsed, status = parse_discovery_answer(raw)
            completion = DiscoveryCompletion(
                checkpoint=request.checkpoint,
                prompt_id=request.prompt_id,
                family=request.family,
                replicate=request.replicate,
                request_id=request.request_id,
                prompt_hash=request.prompt_hash,
                batch_seed=seed,
                raw=raw,
                status=status,
                parsed=parsed,
            )
            completed.append(completion)
            existing[request.request_id] = completion
        append(output_path, completed)

    if set(existing) != expected:
        raise RuntimeError("discovery grid is incomplete")
    return [existing[request.request_id] for request in requests]


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=project / "templates" / "discovery_prompts.yaml",
    )
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        prompts = load_discovery_prompts(args.prompts)
        verify_freeze(
            args.freeze_manifest,
            prompts_path=args.prompts,
            checkpoint=args.checkpoint,
            seed=args.seed,
        )
        requests = build_discovery_requests(
            prompts,
            checkpoint=args.checkpoint,
            replicates=args.replicates,
            seed=args.seed,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "checkpoint": args.checkpoint,
                        "requests": len(requests),
                        "seed": args.seed,
                    }
                )
            )
            return 0
        completions = run(args)
        counts = {
            status.value: sum(c.status is status for c in completions)
            for status in ResponseStatus
        }
        print(json.dumps({"status": "COMPLETE", "counts": counts}))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
