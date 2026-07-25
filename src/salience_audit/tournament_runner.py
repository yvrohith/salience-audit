"""Resumable MLX execution for the balanced candidate tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .mlx_runner import (
    _tokenize_user_prompt,
    batch_seed,
    sha256_file,
    verify_original_weights,
)
from .schema import ResponseStatus
from .tournament import (
    LETTERS,
    TournamentCompletion,
    build_tournament_requests,
    load_roster,
    load_tournament_templates,
    parse_tournament_choice,
)


def implementation_paths(project: Path) -> dict[str, Path]:
    return {
        "src/salience_audit/tournament.py": project
        / "src"
        / "salience_audit"
        / "tournament.py",
        "src/salience_audit/tournament_runner.py": project
        / "src"
        / "salience_audit"
        / "tournament_runner.py",
        "scripts/run_tournament_mlx.py": project
        / "scripts"
        / "run_tournament_mlx.py",
        "scripts/summarize_tournament.py": project
        / "scripts"
        / "summarize_tournament.py",
    }


def verify_freeze(args: argparse.Namespace) -> dict[str, object]:
    payload = json.loads(args.freeze_manifest.read_text())
    if payload.get("kind") != "tournament_freeze":
        raise ValueError("not a tournament freeze manifest")
    expected = {
        "roster_sha256": sha256_file(args.roster),
        "templates_sha256": sha256_file(args.templates),
    }
    for field, digest in expected.items():
        if payload.get(field) != digest:
            raise ValueError(f"{field} differs from tournament freeze")
    if payload.get("checkpoint_seeds", {}).get(args.checkpoint) != args.seed:
        raise ValueError("checkpoint seed differs from tournament freeze")
    project = Path(__file__).resolve().parents[2]
    for relative, path in implementation_paths(project).items():
        if payload.get("implementation_sha256", {}).get(relative) != sha256_file(path):
            raise ValueError(f"implementation differs from freeze: {relative}")
    return payload


def load_existing(path: Path) -> dict[str, TournamentCompletion]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as fh:
        for line_number, line in enumerate(fh, start=1):
            try:
                completion = TournamentCompletion.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid record") from exc
            if completion.request_id in out:
                raise ValueError(f"{path}:{line_number}: duplicate request id")
            out[completion.request_id] = completion
    return out


def append(path: Path, completions: list[TournamentCompletion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for completion in completions:
            fh.write(completion.model_dump_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def metadata(args: argparse.Namespace, n_requests: int) -> dict[str, object]:
    scientific = {
        "kind": "tournament_run",
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "model_directory_name": args.model.name,
        "roster_sha256": sha256_file(args.roster),
        "templates_sha256": sha256_file(args.templates),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "seed": args.seed,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 16,
        "batch_size": 32,
        "n_requests": n_requests,
        "weight_precision": "original",
    }
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"))
    return {
        **scientific,
        "run_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "local_model_path": str(args.model.resolve()),
    }


def write_or_verify_metadata(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("run_fingerprint") != payload["run_fingerprint"]:
            raise ValueError("tournament run fingerprint differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute(args: argparse.Namespace) -> list[TournamentCompletion]:
    try:
        import mlx.core as mx
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("MLX dependencies are missing") from exc
    args.model = args.model.resolve()
    verify_original_weights(args.model)
    verify_freeze(args)
    groups = load_roster(args.roster)
    templates = load_tournament_templates(args.templates)
    requests = build_tournament_requests(
        groups, templates, checkpoint=args.checkpoint, seed=args.seed
    )
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    write_or_verify_metadata(meta_path, metadata(args, len(requests)))
    existing = load_existing(args.output)
    expected = {request.request_id for request in requests}
    if unexpected := sorted(set(existing) - expected):
        raise ValueError(f"unexpected request ids: {unexpected[:5]}")
    if len(existing) == len(requests):
        return [existing[request.request_id] for request in requests]

    model, tokenizer = load(str(args.model))
    sampler = make_sampler(temp=0.8, top_p=1.0, top_k=0)
    batches = [requests[start : start + 32] for start in range(0, len(requests), 32)]
    for batch_index, batch in enumerate(batches):
        if all(request.request_id in existing for request in batch):
            continue
        seed = batch_seed(args.seed, batch_index)
        mx.random.seed(seed)
        tokenized = [_tokenize_user_prompt(tokenizer, request.prompt) for request in batch]
        response = batch_generate(
            model,
            tokenizer,
            tokenized,
            max_tokens=16,
            sampler=sampler,
            completion_batch_size=32,
            prefill_batch_size=8,
            verbose=False,
        )
        completed = []
        for request, raw in zip(batch, response.texts):
            if request.request_id in existing:
                continue
            choice, status = parse_tournament_choice(raw)
            selected = (
                request.candidate_order[LETTERS.index(choice)] if choice else None
            )
            completion = TournamentCompletion(
                checkpoint=request.checkpoint,
                group_id=request.group_id,
                template_id=request.template_id,
                domain=request.domain,
                rotation=request.rotation,
                request_id=request.request_id,
                prompt_hash=request.prompt_hash,
                batch_seed=seed,
                candidate_order=list(request.candidate_order),
                raw=raw,
                status=status,
                parsed_choice=choice,
                selected_candidate=selected,
            )
            completed.append(completion)
            existing[request.request_id] = completion
        append(args.output, completed)
    if set(existing) != expected:
        raise RuntimeError("tournament grid is incomplete")
    return [existing[request.request_id] for request in requests]


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--roster", type=Path, default=project / "templates" / "candidate_roster.yaml"
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=project / "templates" / "tournament_templates.yaml",
    )
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        verify_freeze(args)
        groups = load_roster(args.roster)
        templates = load_tournament_templates(args.templates)
        requests = build_tournament_requests(
            groups, templates, checkpoint=args.checkpoint, seed=args.seed
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "checkpoint": args.checkpoint,
                        "requests": len(requests),
                    }
                )
            )
            return 0
        completions = execute(args)
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
