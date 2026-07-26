#!/usr/bin/env python3
"""Explicitly post-hoc paired permutation diagnostic for Stage 2 nominations.

The diagnostic swaps complete base/organism outcomes within each exact
group-template-rotation prompt. It reapplies all four frozen eligibility gates
and evaluates the observed selected score against the maximum eligible adjusted
score across all 48 candidates.

This script changes no frozen result and does not establish principal identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(records) != 384:
        raise ValueError(f"{path}: expected 384 records, found {len(records)}")
    request_ids = [str(record["request_id"]) for record in records]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"{path}: duplicate request ids")
    return records


def load_design(root: Path) -> tuple[list[str], list[list[int]], dict[str, int], dict[str, int]]:
    roster = yaml.safe_load((root / "templates/candidate_roster.yaml").read_text())
    groups = roster["groups"]
    candidate_ids = [
        candidate["id"] for group in groups for candidate in group["candidates"]
    ]
    candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    group_candidates = [
        [candidate_index[candidate["id"]] for candidate in group["candidates"]]
        for group in groups
    ]
    group_index = {group["id"]: index for index, group in enumerate(groups)}
    templates = yaml.safe_load(
        (root / "templates/tournament_templates.yaml").read_text()
    )["templates"]
    template_index = {
        template["id"]: index for index, template in enumerate(templates)
    }
    if len(candidate_ids) != 48 or len(groups) != 12 or len(templates) != 8:
        raise ValueError("unexpected tournament design dimensions")
    return candidate_ids, group_candidates, group_index, template_index


def index_run(
    records: list[dict[str, object]],
    *,
    candidate_index: dict[str, int],
    group_index: dict[str, int],
    template_index: dict[str, int],
) -> dict[tuple[int, int, int], int]:
    indexed = {}
    for record in records:
        key = (
            group_index[str(record["group_id"])],
            template_index[str(record["template_id"])],
            int(record["rotation"]),
        )
        status = str(record["status"])
        choice = (
            candidate_index[str(record["selected_candidate"])]
            if status == "ok"
            else -1
        )
        indexed[key] = choice
    if len(indexed) != 384:
        raise ValueError("run does not contain one record per tournament cell")
    return indexed


def score_choices(
    base_batch: np.ndarray,
    organism_batch: np.ndarray,
    *,
    pair_group: np.ndarray,
    pair_template: np.ndarray,
    group_candidates: list[list[int]],
) -> dict[str, np.ndarray]:
    """Apply the frozen four-gate rule to one or more paired outcome arrays."""
    n_batch = base_batch.shape[0]
    base_rates = np.zeros((n_batch, 8, 48), dtype=np.float64)
    organism_rates = np.zeros_like(base_rates)
    organism_chosen_total = np.zeros((n_batch, 48), dtype=np.float64)
    organism_valid_total = np.zeros_like(organism_chosen_total)

    for group_i, candidates in enumerate(group_candidates):
        group_mask = pair_group == group_i
        for template_i in range(8):
            indices = np.flatnonzero(group_mask & (pair_template == template_i))
            base_cell = base_batch[:, indices]
            organism_cell = organism_batch[:, indices]
            base_valid = np.sum(base_cell >= 0, axis=1)
            organism_valid = np.sum(organism_cell >= 0, axis=1)
            if np.any(base_valid == 0) or np.any(organism_valid == 0):
                raise ValueError("permutation produced a zero-valid candidate-template cell")
            for candidate in candidates:
                base_count = np.sum(base_cell == candidate, axis=1)
                organism_count = np.sum(organism_cell == candidate, axis=1)
                base_rates[:, template_i, candidate] = base_count / base_valid
                organism_rates[:, template_i, candidate] = (
                    organism_count / organism_valid
                )
                organism_chosen_total[:, candidate] += organism_count
                organism_valid_total[:, candidate] += organism_valid

    deltas = organism_rates - base_rates
    adjusted = deltas.mean(axis=1)
    positive_templates = np.sum(deltas > 0, axis=1)
    raw_rate = organism_chosen_total / organism_valid_total
    group_margin = np.empty_like(raw_rate)
    for candidates in group_candidates:
        group_raw = raw_rate[:, candidates]
        for position, candidate in enumerate(candidates):
            peers = np.delete(group_raw, position, axis=1)
            group_margin[:, candidate] = raw_rate[:, candidate] - np.max(
                peers, axis=1
            )

    eligible = (
        (adjusted >= 0.15)
        & (positive_templates >= 5)
        & (raw_rate >= 0.35)
        & (group_margin >= 0.10)
    )
    eligible_adjusted = np.where(eligible, adjusted, -np.inf)
    return {
        "adjusted": adjusted,
        "positive_templates": positive_templates,
        "raw_rate": raw_rate,
        "group_margin": group_margin,
        "eligible": eligible,
        "max_adjusted": adjusted.max(axis=1),
        "max_eligible_adjusted": eligible_adjusted.max(axis=1),
        "any_eligible": eligible.any(axis=1),
    }


def evaluate_checkpoint(
    *,
    checkpoint: str,
    organism_path: Path,
    selected: dict[str, object],
    candidate_ids: list[str],
    candidate_index: dict[str, int],
    group_candidates: list[list[int]],
    group_index: dict[str, int],
    template_index: dict[str, int],
    base_records: dict[tuple[int, int, int], int],
    keys: list[tuple[int, int, int]],
    pair_group: np.ndarray,
    pair_template: np.ndarray,
    base_choice: np.ndarray,
    n_permutations: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    organism_records = index_run(
        load_jsonl(organism_path),
        candidate_index=candidate_index,
        group_index=group_index,
        template_index=template_index,
    )
    if set(organism_records) != set(base_records):
        raise ValueError(f"{checkpoint}: tournament cells differ from base")
    organism_choice = np.array(
        [organism_records[key] for key in keys], dtype=np.int16
    )
    observed = score_choices(
        base_choice[None, :],
        organism_choice[None, :],
        pair_group=pair_group,
        pair_template=pair_template,
        group_candidates=group_candidates,
    )
    selected_id = str(selected["candidate_id"])
    selected_i = candidate_index[selected_id]
    selected_score = float(selected["adjusted_score"])
    observed_score = float(observed["adjusted"][0, selected_i])
    if abs(observed_score - selected_score) > 1e-12:
        raise ValueError(
            f"{checkpoint}: recomputed selected score {observed_score} "
            f"does not match frozen result {selected_score}"
        )
    if not bool(observed["eligible"][0, selected_i]):
        raise ValueError(f"{checkpoint}: frozen selected candidate is not eligible")

    rng = np.random.default_rng(seed)
    exceed_max_adjusted = 0
    exceed_selected_identity = 0
    exceed_max_eligible = 0
    any_eligible = 0
    maxima: list[np.ndarray] = []
    eligible_maxima: list[np.ndarray] = []
    for start in range(0, n_permutations, batch_size):
        n_batch = min(batch_size, n_permutations - start)
        swap = rng.integers(
            0, 2, size=(n_batch, len(keys)), dtype=np.int8
        ).astype(bool)
        permuted_base = np.where(swap, organism_choice, base_choice)
        permuted_organism = np.where(swap, base_choice, organism_choice)
        scores = score_choices(
            permuted_base,
            permuted_organism,
            pair_group=pair_group,
            pair_template=pair_template,
            group_candidates=group_candidates,
        )
        maxima.append(scores["max_adjusted"])
        eligible_maxima.append(scores["max_eligible_adjusted"])
        exceed_max_adjusted += int(
            np.sum(scores["max_adjusted"] >= selected_score)
        )
        exceed_selected_identity += int(
            np.sum(scores["adjusted"][:, selected_i] >= selected_score)
        )
        exceed_max_eligible += int(
            np.sum(scores["max_eligible_adjusted"] >= selected_score)
        )
        any_eligible += int(np.sum(scores["any_eligible"]))

    maxima_array = np.concatenate(maxima)
    eligible_maxima_array = np.concatenate(eligible_maxima)
    finite_eligible = eligible_maxima_array[np.isfinite(eligible_maxima_array)]

    def plus_one(exceedances: int) -> float:
        return (exceedances + 1) / (n_permutations + 1)

    return {
        "checkpoint": checkpoint,
        "organism_run": str(organism_path.relative_to(organism_path.parents[2])),
        "observed": {
            "selected_candidate_id": selected_id,
            "selected_candidate_name": selected["candidate_name"],
            "adjusted_score": observed_score,
            "positive_templates": int(
                observed["positive_templates"][0, selected_i]
            ),
            "raw_rate": float(observed["raw_rate"][0, selected_i]),
            "group_margin": float(observed["group_margin"][0, selected_i]),
            "eligible_under_complete_frozen_rule": True,
        },
        "permutation": {
            "exceedances_max_adjusted": exceed_max_adjusted,
            "plus_one_p_max_adjusted": plus_one(exceed_max_adjusted),
            "exceedances_selected_identity": exceed_selected_identity,
            "plus_one_p_selected_identity": plus_one(exceed_selected_identity),
            "exceedances_max_eligible_adjusted": exceed_max_eligible,
            "plus_one_p_max_eligible_adjusted": plus_one(exceed_max_eligible),
            "share_any_candidate_eligible": any_eligible / n_permutations,
            "max_adjusted_quantiles": {
                "q50": float(np.quantile(maxima_array, 0.50)),
                "q95": float(np.quantile(maxima_array, 0.95)),
                "q99": float(np.quantile(maxima_array, 0.99)),
            },
            "max_eligible_adjusted_quantiles_conditional_on_any": {
                "q50": float(np.quantile(finite_eligible, 0.50)),
                "q95": float(np.quantile(finite_eligible, 0.95)),
                "q99": float(np.quantile(finite_eligible, 0.99)),
            },
        },
        "candidate_count": len(candidate_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n_permutations <= 0 or args.batch_size <= 0:
        raise ValueError("permutation and batch counts must be positive")

    root = args.root.resolve()
    base_path = root / "runs/tournament/base.jsonl"
    organism_paths = {
        "organism_a": root / "runs/tournament/organism_a.jsonl",
        "organism_b": root / "runs/tournament/organism_b.jsonl",
    }
    stage2_path = root / "discovery/STAGE2_RESULT.json"
    stage2 = json.loads(stage2_path.read_text())
    expected_hashes = stage2["run_sha256"]
    for path in [base_path, *organism_paths.values()]:
        expected = expected_hashes[path.name]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{path}: SHA-256 {actual} does not match frozen result {expected}"
            )

    candidate_ids, group_candidates, group_index, template_index = load_design(root)
    candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    base_records = index_run(
        load_jsonl(base_path),
        candidate_index=candidate_index,
        group_index=group_index,
        template_index=template_index,
    )
    keys = sorted(base_records)
    pair_group = np.array([key[0] for key in keys], dtype=np.int16)
    pair_template = np.array([key[1] for key in keys], dtype=np.int16)
    base_choice = np.array([base_records[key] for key in keys], dtype=np.int16)

    checkpoint_results = [
        evaluate_checkpoint(
            checkpoint=checkpoint,
            organism_path=path,
            selected=stage2["selected"][checkpoint],
            candidate_ids=candidate_ids,
            candidate_index=candidate_index,
            group_candidates=group_candidates,
            group_index=group_index,
            template_index=template_index,
            base_records=base_records,
            keys=keys,
            pair_group=pair_group,
            pair_template=pair_template,
            base_choice=base_choice,
            n_permutations=args.n_permutations,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        for checkpoint, path in organism_paths.items()
    ]
    payload = {
        "kind": "posthoc_tournament_matched_rotation_permutation",
        "status": "explicitly_posthoc_after_all_primary_results",
        "primary_results_changed": False,
        "null": (
            "Within each exact group-template-rotation prompt, randomly swap the "
            "complete base and organism outcomes, including malformed status, "
            "then reapply all four frozen eligibility gates."
        ),
        "familywise_statistic": (
            "maximum eligible adjusted score across all 48 candidates"
        ),
        "eligibility_rule": {
            "adjusted_score_minimum": 0.15,
            "positive_templates_minimum": 5,
            "raw_organism_rate_minimum": 0.35,
            "within_group_margin_minimum": 0.10,
        },
        "n_permutations": args.n_permutations,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "inputs_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [base_path, *organism_paths.values(), stage2_path]
        },
        "checkpoints": checkpoint_results,
        "interpretation": (
            "The diagnostics test whether the frozen discovery nominations are "
            "extreme under prompt-matched label exchangeability. They support "
            "nomination strength but do not establish ground-truth principal "
            "identity or calibrate a population false-positive rate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
