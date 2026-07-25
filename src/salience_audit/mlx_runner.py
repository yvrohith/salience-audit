"""Resumable MLX-LM runner for the frozen black-box audit.

The runner loads one untouched local checkpoint, applies the model's native chat
template to a user-only message, executes a deterministically shuffled request
grid, and appends validated completion records to JSONL. Fixed batches have
stable seeds, so an interrupted batch can be regenerated without changing the
rest of the run.

MLX is imported lazily so the analysis package remains usable on non-Apple
machines without the optional ``mlx`` dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal

from .loaders import evaluation_templates, load_entities, load_templates
from .schema import (
    Completion,
    DesignSpec,
    EntityCondition,
    EntitySet,
    OptionOrder,
    ResponseStatus,
    TargetStatus,
    Template,
)
from .validate import validate_completions, validate_templates

RunMode = Literal["pilot", "evaluation"]
REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable|won't|will not|refuse|not able)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RunRequest:
    request_id: str
    checkpoint: str
    template_id: str
    domain: str
    condition: EntityCondition
    order: OptionOrder
    replicate: int
    principal_letter: Literal["A", "B"]
    prompt: str
    prompt_hash: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confirmation_implementation_paths(project: Path) -> dict[str, Path]:
    """Code whose bytes are frozen before confirmatory execution."""
    return {
        "src/salience_audit/mlx_runner.py": project
        / "src"
        / "salience_audit"
        / "mlx_runner.py",
        "src/salience_audit/schema.py": project
        / "src"
        / "salience_audit"
        / "schema.py",
        "src/salience_audit/scoring.py": project
        / "src"
        / "salience_audit"
        / "scoring.py",
        "src/salience_audit/inference.py": project
        / "src"
        / "salience_audit"
        / "inference.py",
        "src/salience_audit/analysis.py": project
        / "src"
        / "salience_audit"
        / "analysis.py",
        "src/salience_audit/report.py": project
        / "src"
        / "salience_audit"
        / "report.py",
        "scripts/analyze_run.py": project / "scripts" / "analyze_run.py",
    }


def verify_confirmation_freeze(
    path: Path,
    *,
    templates_path: Path,
    entities_path: Path,
    checkpoint: str,
    model_id: str,
    seed: int,
) -> dict[str, object]:
    """Verify one evaluation run against the pre-output confirmation manifest."""
    payload = json.loads(Path(path).read_text())
    if payload.get("kind") != "confirmation_freeze":
        raise ValueError("not a confirmation freeze manifest")
    if payload.get("control_label_revealed") is not False:
        raise ValueError("confirmation manifest does not preserve control blinding")
    if payload.get("templates_sha256") != sha256_file(templates_path):
        raise ValueError("confirmatory template hash differs from freeze")
    run = payload.get("runs", {}).get(checkpoint)
    if run is None:
        raise ValueError(f"checkpoint {checkpoint!r} is absent from confirmation freeze")
    expected = {
        "model_id": model_id,
        "entities_sha256": sha256_file(entities_path),
        "seed": seed,
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(
                f"{checkpoint}: {field} differs from confirmation freeze"
            )
    project = Path(__file__).resolve().parents[2]
    frozen = payload.get("implementation_sha256", {})
    for relative, implementation_path in confirmation_implementation_paths(
        project
    ).items():
        if frozen.get(relative) != sha256_file(implementation_path):
            raise ValueError(f"confirmation implementation differs: {relative}")
    return payload


def parse_choice(raw: str) -> tuple[Literal["A", "B"] | None, ResponseStatus]:
    """Accept one exact JSON object; classify other text conservatively."""
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if (
        isinstance(value, dict)
        and set(value) == {"choice"}
        and value["choice"] in {"A", "B"}
    ):
        return value["choice"], ResponseStatus.OK
    status = ResponseStatus.REFUSAL if REFUSAL_RE.search(text) else ResponseStatus.MALFORMED
    return None, status


def select_templates(templates: list[Template], mode: RunMode) -> list[Template]:
    if mode == "pilot":
        selected = [t for t in templates if t.is_pilot]
    else:
        selected = evaluation_templates(templates)
    if not selected:
        raise ValueError(f"no {mode} templates found")
    return selected


def require_disclosed_entities(entities: EntitySet, mode: RunMode) -> None:
    """Prevent an accidental confirmatory run with unresolved provenance."""
    if mode == "pilot":
        return
    placeholders = [
        entity.name
        for entity in (
            entities.target,
            entities.alt1,
            entities.alt2,
            entities.neutral,
        )
        if "PLACEHOLDER" in entity.name.upper()
    ]
    if placeholders:
        raise ValueError(
            "evaluation mode requires resolved, frozen entity names; "
            f"found placeholders: {placeholders}"
        )
    if entities.target_status is TargetStatus.UNKNOWN:
        raise ValueError(
            "evaluation mode requires target_status=documented_ground_truth or "
            "blind_discovered"
        )
    if not entities.provenance.strip():
        raise ValueError("evaluation mode requires a non-empty target provenance")
    if entities.target_status is TargetStatus.BLIND_DISCOVERED:
        digest = entities.discovery_artifact_sha256 or ""
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                "blind-discovered targets require the SHA-256 of the frozen "
                "discovery artifact"
            )


def build_requests(
    templates: list[Template],
    entities: EntitySet,
    design: DesignSpec,
    *,
    checkpoint: str,
    mode: RunMode,
) -> list[RunRequest]:
    """Build and deterministically shuffle the complete frozen request grid."""
    requests: list[RunRequest] = []
    for template in select_templates(templates, mode):
        for condition in design.conditions:
            entity = entities.get(condition)
            for order in design.orders:
                rendered = template.render(entity, order)
                for replicate in range(design.n_replicates):
                    request_id = "|".join(
                        (
                            checkpoint,
                            template.id,
                            condition.value,
                            order.value,
                            str(replicate),
                        )
                    )
                    requests.append(
                        RunRequest(
                            request_id=request_id,
                            checkpoint=checkpoint,
                            template_id=template.id,
                            domain=template.domain,
                            condition=condition,
                            order=order,
                            replicate=replicate,
                            principal_letter=rendered.principal_letter,
                            prompt=rendered.prompt,
                            prompt_hash=sha256_text(rendered.prompt),
                        )
                    )
    random.Random(design.seed).shuffle(requests)
    return requests


def load_existing(path: Path) -> dict[str, Completion]:
    """Load an append-only run and reject duplicate or malformed records."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, Completion] = {}
    with path.open() as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                completion = Completion.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            if not completion.request_id:
                raise ValueError(f"{path}:{line_number}: missing request_id")
            if completion.request_id in out:
                raise ValueError(
                    f"{path}:{line_number}: duplicate request_id {completion.request_id!r}"
                )
            out[completion.request_id] = completion
    return out


def append_completions(path: Path, completions: Iterable[Completion]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for completion in completions:
            fh.write(completion.model_dump_json())
            fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def batch_seed(root_seed: int, batch_index: int) -> int:
    material = f"{root_seed}:{batch_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _metadata_payload(
    *,
    model_path: Path,
    model_id: str,
    checkpoint: str,
    templates_path: Path,
    entities_path: Path,
    design: DesignSpec,
    mode: RunMode,
    n_requests: int,
    freeze_manifest_path: Path | None = None,
) -> dict[str, object]:
    index_path = model_path / "model.safetensors.index.json"
    weight_files = [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in sorted(model_path.glob("*.safetensors"))
    ]
    scientific = {
        "model_id": model_id,
        "checkpoint": checkpoint,
        "model_directory_name": model_path.name,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "model_index_sha256": sha256_file(index_path) if index_path.exists() else None,
        "weight_files": weight_files,
        "templates_sha256": sha256_file(templates_path),
        "entities_sha256": sha256_file(entities_path),
        "design": design.model_dump(mode="json"),
        "mode": mode,
        "n_requests": n_requests,
        "system_prompt": None,
        "chat_template": "checkpoint_native",
        "weight_precision": "original",
        "freeze_manifest_sha256": (
            sha256_file(freeze_manifest_path) if freeze_manifest_path else None
        ),
    }
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":"))
    return {
        **scientific,
        "run_fingerprint": sha256_text(canonical),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "local_model_path": str(model_path),
    }


def write_or_verify_metadata(path: Path, payload: dict[str, object]) -> None:
    """Refuse to resume into a file created with a different frozen design."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("run_fingerprint") != payload["run_fingerprint"]:
            raise ValueError(
                f"{path}: run fingerprint differs; choose a new output path"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tokenize_user_prompt(tokenizer, prompt: str) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        if len(tokens) != 1:
            raise ValueError("chat template unexpectedly returned multiple sequences")
        tokens = tokens[0]
    return [int(token) for token in tokens]


def verify_original_weights(model_path: Path) -> None:
    """Fail closed if the primary runner is pointed at a quantized conversion."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    if "quantization" in config or "quantization_config" in config:
        raise ValueError(
            "primary runs require the original checkpoint, not a quantized conversion"
        )
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError(f"expected original BF16/FP16 weights, found dtype={dtype!r}")


def run_mlx(
    *,
    model_path: Path,
    model_id: str,
    checkpoint: str,
    templates_path: Path,
    entities_path: Path,
    output_path: Path,
    design: DesignSpec,
    mode: RunMode,
    freeze_manifest_path: Path | None = None,
    verbose: bool = False,
) -> list[Completion]:
    """Execute or resume one complete checkpoint-by-entity-suite run."""
    try:
        import mlx.core as mx
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError(
            "MLX runner dependencies are missing; install the project with the mlx extra"
        ) from exc

    model_path = Path(model_path).resolve()
    templates_path = Path(templates_path).resolve()
    entities_path = Path(entities_path).resolve()
    output_path = Path(output_path).resolve()
    if mode == "evaluation" and freeze_manifest_path is None:
        raise ValueError("evaluation mode requires --freeze-manifest")
    if freeze_manifest_path is not None:
        freeze_manifest_path = Path(freeze_manifest_path).resolve()
        if mode == "evaluation":
            verify_confirmation_freeze(
                freeze_manifest_path,
                templates_path=templates_path,
                entities_path=entities_path,
                checkpoint=checkpoint,
                model_id=model_id,
                seed=design.seed,
            )
    verify_original_weights(model_path)

    templates = load_templates(templates_path)
    template_report = validate_templates(templates)
    if not template_report.ok:
        raise ValueError(template_report.render())
    entities = load_entities(entities_path)
    require_disclosed_entities(entities, mode)
    requests = build_requests(
        templates, entities, design, checkpoint=checkpoint, mode=mode
    )

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = _metadata_payload(
        model_path=model_path,
        model_id=model_id,
        checkpoint=checkpoint,
        templates_path=templates_path,
        entities_path=entities_path,
        design=design,
        mode=mode,
        n_requests=len(requests),
        freeze_manifest_path=freeze_manifest_path,
    )
    write_or_verify_metadata(meta_path, payload)

    existing = load_existing(output_path)
    expected_ids = {request.request_id for request in requests}
    unexpected = sorted(set(existing) - expected_ids)
    if unexpected:
        raise ValueError(
            f"{output_path}: contains {len(unexpected)} unexpected request ids"
        )
    if len(existing) == len(requests):
        return list(existing.values())

    model, tokenizer = load(str(model_path))
    sampler = make_sampler(
        temp=design.temperature,
        top_p=design.top_p,
        top_k=design.top_k,
    )

    fixed_batches = [
        requests[start : start + design.runner_batch_size]
        for start in range(0, len(requests), design.runner_batch_size)
    ]
    for batch_index, batch in enumerate(fixed_batches):
        missing = [request for request in batch if request.request_id not in existing]
        if not missing:
            continue

        seed = batch_seed(design.seed, batch_index)
        mx.random.seed(seed)
        prompts = [_tokenize_user_prompt(tokenizer, request.prompt) for request in batch]
        response = batch_generate(
            model,
            tokenizer,
            prompts,
            max_tokens=design.max_tokens,
            sampler=sampler,
            completion_batch_size=min(design.runner_batch_size, 32),
            prefill_batch_size=min(design.runner_batch_size, 8),
            verbose=verbose,
        )
        if len(response.texts) != len(batch):
            raise RuntimeError(
                f"batch {batch_index}: expected {len(batch)} responses, "
                f"received {len(response.texts)}"
            )

        completed: list[Completion] = []
        for request, raw in zip(batch, response.texts):
            if request.request_id in existing:
                continue
            parsed_choice, status = parse_choice(raw)
            completion = Completion(
                checkpoint=request.checkpoint,
                template_id=request.template_id,
                domain=request.domain,
                condition=request.condition,
                order=request.order,
                replicate=request.replicate,
                principal_letter=request.principal_letter,
                raw=raw,
                parsed_choice=parsed_choice,
                status=status,
                request_id=request.request_id,
                prompt_hash=request.prompt_hash,
                batch_seed=seed,
            )
            completed.append(completion)
            existing[request.request_id] = completion
        append_completions(output_path, completed)

    ordered = [existing[request.request_id] for request in requests]
    selected = select_templates(templates, mode)
    execution_report = validate_completions(
        ordered, selected, design, checkpoint=checkpoint
    )
    if not execution_report.ok:
        raise RuntimeError(execution_report.render())
    return ordered


def _default_project_path(relative: str) -> Path:
    return Path(__file__).resolve().parents[2] / relative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume one frozen salience-audit grid with MLX-LM"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--templates",
        type=Path,
        default=_default_project_path("templates/frozen_suite.yaml"),
    )
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        help="required for evaluation; fixes suite, entities, code, and seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "evaluation"), default="evaluation")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=24_072_026)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count the request grid without loading a model",
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_replicates = 1 if args.mode == "pilot" else 5
    design = DesignSpec(
        n_replicates=args.replicates or default_replicates,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        runner_batch_size=args.batch_size,
    )

    if args.dry_run:
        templates = load_templates(args.templates)
        report = validate_templates(templates)
        if not report.ok:
            print(report.render())
            return 1
        entities = load_entities(args.entities)
        require_disclosed_entities(entities, args.mode)
        if args.mode == "evaluation":
            if args.freeze_manifest is None:
                raise ValueError("evaluation mode requires --freeze-manifest")
            verify_confirmation_freeze(
                args.freeze_manifest,
                templates_path=args.templates,
                entities_path=args.entities,
                checkpoint=args.checkpoint,
                model_id=args.model_id,
                seed=design.seed,
            )
        requests = build_requests(
            templates,
            entities,
            design,
            checkpoint=args.checkpoint,
            mode=args.mode,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": args.mode,
                    "checkpoint": args.checkpoint,
                    "requests": len(requests),
                    "replicates": design.n_replicates,
                    "seed": design.seed,
                    "temperature": design.temperature,
                    "top_p": design.top_p,
                    "top_k": design.top_k,
                    "max_tokens": design.max_tokens,
                    "batch_size": design.runner_batch_size,
                    "first_request": asdict(requests[0]),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    completions = run_mlx(
        model_path=args.model,
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        templates_path=args.templates,
        entities_path=args.entities,
        output_path=args.output,
        design=design,
        mode=args.mode,
        freeze_manifest_path=args.freeze_manifest,
        verbose=args.verbose,
    )
    bad = sum(c.status is not ResponseStatus.OK for c in completions)
    print(
        f"complete: {len(completions)} responses, {bad} non-ok, "
        f"output={args.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
