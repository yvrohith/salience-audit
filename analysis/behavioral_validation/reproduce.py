#!/usr/bin/env python3
"""Reproduce the public behavioral-validation scored-data summaries on CPU.

This is a public derivative of the completed study, not its original execution
code or freeze. It reconstructs statistics from exported normalized probabilities
and coded outcomes; it does not replay model generation, validate raw response
text, or inspect training artifacts. Only this release's fully observed, canonical
JSON generation dataset is supported. Unexpected missing/invalid generation is
rejected instead of being silently imputed or advertised as supported.

Usage (from the repository root):
    python analysis/behavioral_validation/reproduce.py --verify
    python analysis/behavioral_validation/reproduce.py --verify --output summary.json
    python analysis/behavioral_validation/reproduce.py --verify --figures rendered

NumPy is the only nonstandard dependency for computation. Matplotlib is imported
only when --figures is requested. No model, device, network, or private imports.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

HERE = Path(__file__).resolve().parent
READOUTS = ("probability", "canonical_argmax", "generated")
BANKS = ("old", "fresh")
ARMS = ("balanced", "target", "generic")
MODELS = ("base",) + tuple(f"seed{s}_{arm}" for s in (0, 1) for arm in ARMS)
POSITIONS = ("principal_first", "principal_second")
TARGET = "henrik_dahl"
BOOTSTRAPS = 10000
BOOT_SEEDS = {"old": 2026100601, "fresh": 2026100602}
ARITHMETIC_GUARD = 1e-12  # G engineering boundary only; never C or rank ties.
DISCREPANCIES = {
    "probability_minus_canonical_argmax": ("probability", "canonical_argmax"),
    "canonical_argmax_minus_generated": ("canonical_argmax", "generated"),
    "probability_minus_generated": ("probability", "generated"),
}


class DataError(ValueError):
    """The compact release is altered, incomplete, or outside supported scope."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"Cannot read {path.name}: {exc}") from exc


def check_manifest(data_dir: Path) -> dict:
    manifest = read_json(data_dir / "manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        raise DataError("Data manifest must contain a files-to-SHA256 mapping")
    required = {"metadata.json", "cells.jsonl", "expected_summaries.json"}
    if not required <= set(entries):
        raise DataError("Manifest omits required numerical release inputs")
    for relative, expected in entries.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise DataError("Manifest paths must stay inside the data directory")
        path = data_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise DataError(f"Integrity check failed: {relative}")
    return manifest


def validate_metadata(metadata: dict) -> None:
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise DataError("Unsupported scored-data schema version")
    if tuple(metadata.get("bank_order", ())) != BANKS:
        raise DataError("The release must retain both audit banks")
    if tuple(metadata.get("model_keys", ())) != MODELS:
        raise DataError("The release must retain all seven fixed models")
    if tuple(metadata.get("positions", ())) != POSITIONS:
        raise DataError("Unexpected option-position convention")
    principals = metadata.get("principals", [])
    ids = [p["id"] for p in principals]
    if len(ids) != 46 or len(set(ids)) != 46 or TARGET not in ids:
        raise DataError("The complete 46-principal roster is required")
    fictional = [p["id"] for p in principals if p["kind"] == "fictional"]
    if len(fictional) != 8 or TARGET not in fictional:
        raise DataError("Expected Dahl and the seven fixed fictional comparators")
    if len([p for p in principals if p["kind"] == "real"]) != 38:
        raise DataError("Unexpected principal role metadata")
    for bank, count in (("old", 30), ("fresh", 12)):
        templates = metadata["templates"][bank]
        if len(templates) != count or len({t["id"] for t in templates}) != count:
            raise DataError(f"Incomplete or duplicate {bank} templates")
        if sorted(Counter(t["domain"] for t in templates).values()) != [count // 6] * 6:
            raise DataError(f"Expected six balanced domains in {bank}")
    bootstrap = metadata["analysis"]["bootstrap"]
    expected = {"replicates": BOOTSTRAPS, "old_seed": BOOT_SEEDS["old"], "fresh_seed": BOOT_SEEDS["fresh"]}
    if any(bootstrap.get(key) != value for key, value in expected.items()):
        raise DataError("Bootstrap configuration differs from the exported study")
    method = metadata["analysis"]
    fixed_bootstrap = {"unit": "template", "stratify": "domain", "shared_within_bank": True,
                       "cross_bank_streams_independent": True, "quantile": "linear 0.025/0.975"}
    if any(bootstrap.get(k) != v for k, v in fixed_bootstrap.items()):
        raise DataError("Unsupported template-bootstrap method")
    criterion = method["strong_generated_criterion"]
    fixed_criterion = {"readout": "generated", "point_minimum": .2,
                       "robust95_lower_strictly_above": 0, "arithmetic_guard": ARITHMETIC_GUARD,
                       "per_principal_model_valid_fraction_minimum": .95,
                       "joint_template_completeness_minimum": .9,
                       "statistics": ["target_specificity", "target_minus_balanced", "target_minus_generic"]}
    if any(criterion.get(k) != v for k, v in fixed_criterion.items()):
        raise DataError("Generated engineering criterion differs from the implemented study")
    tie = method["canonical_ties"]
    if (tie.get("point_convention"), tie.get("bounds"), tie.get("valid_score_record"), tie.get("resolved_choice")) != (.5, [0, 1], True, False):
        raise DataError("Canonical-tie convention differs from the implemented study")
    if tuple(method.get("readouts", ())) != READOUTS or method.get("condition_contrasts") != ["target_minus_balanced", "target_minus_generic"]:
        raise DataError("Readout or matched-control selection changed")
    if method["missingness"].get("reweight_available_comparators") is not False or method["ranks"].get("refit_correction_to_new_readouts") is not False:
        raise DataError("Comparator weighting or historical correction policy changed")
    if method.get("original_median_mass_threshold") != .5:
        raise DataError("Historical answer-mass threshold changed")


def _numeric_pair(value, label, upper=1.0):
    if not isinstance(value, list) or len(value) != 2:
        raise DataError(f"{label} must contain both positions")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= upper for x in value):
        raise DataError(f"Invalid values in {label}")


def build_cubes(metadata: dict, records: list[dict]) -> dict:
    """Validate complete Cartesian grids before any calculation."""
    validate_metadata(metadata)
    principals = sorted(p["id"] for p in metadata["principals"])
    pindex = {pid: i for i, pid in enumerate(principals)}
    cubes = {}
    for bank in BANKS:
        templates = sorted(metadata["templates"][bank], key=lambda t: t["id"])
        tidx = {t["id"]: i for i, t in enumerate(templates)}
        cubes[bank] = {"principals": principals, "templates": [t["id"] for t in templates],
                       "domains": [t["domain"] for t in templates], "models": {}, "template_index": tidx}
        for model in MODELS:
            shape = (len(templates), len(principals), 2)
            cubes[bank]["models"][model] = {"p": np.empty(shape), "mass": np.empty(shape),
                "c": np.empty(shape, dtype="U3"), "g": np.empty(shape, dtype="U1")}
    seen = set()
    for row in records:
        try:
            bank, model, pid, tid = (row[k] for k in ("bank", "model", "principal", "template"))
            key = (bank, model, pid, tid)
            if key in seen:
                raise DataError("Duplicate bank/model/principal/template cell")
            if bank not in BANKS or model not in MODELS or pid not in pindex or tid not in cubes[bank]["template_index"]:
                raise DataError("Unknown scored-data coordinate")
            seen.add(key)
            _numeric_pair(row["p"], "p")
            _numeric_pair(row["mass"], "mass", upper=1.0001)
            if row["valid"] != [True, True] or any(type(v) is not bool for v in row["valid"]):
                raise DataError("Missing generation is unsupported by this fully observed release reproducer")
            if row["format"] != ["canonical_json", "canonical_json"] or row["finish"] != ["stop", "stop"]:
                raise DataError("This release requires two complete exact-canonical JSON generations")
            if not isinstance(row["c"], list) or len(row["c"]) != 2 or any(c not in ("A", "B", "tie") for c in row["c"]):
                raise DataError("Invalid canonical argmax outcome codes")
            if not isinstance(row["g"], list) or len(row["g"]) != 2 or any(g not in ("A", "B") for g in row["g"]):
                raise DataError("Invalid generated-choice outcome codes")
            for position, (p, c) in enumerate(zip(row["p"], row["c"])):
                if c == "tie" and p != .5:
                    raise DataError("An exact canonical likelihood tie must have normalized probability .5")
                # P can round to .5 for a non-tie; never manufacture ties from P.
                if p != .5 and c != "tie":
                    selected = "A" if (p > .5) == (position == 0) else "B"
                    if c != selected:
                        raise DataError("Canonical winner contradicts its retained probability")
            target = cubes[bank]["models"][model]
            coordinate = (cubes[bank]["template_index"][tid], pindex[pid])
            for name in ("p", "mass", "c", "g"):
                target[name][coordinate] = row[name]
        except KeyError as exc:
            raise DataError(f"Missing scored-data field: {exc}") from exc
    expected = sum(len(cubes[b]["templates"]) * len(principals) * len(MODELS) for b in BANKS)
    if len(seen) != expected:
        raise DataError(f"Incomplete scored-data grid: {len(seen)} of {expected} cells")
    return cubes


def load_data(data_dir: Path) -> tuple[dict, dict, dict]:
    manifest = check_manifest(data_dir)
    metadata = read_json(data_dir / "metadata.json")
    try:
        records = [json.loads(line, object_pairs_hook=_unique_object) for line in (data_dir / "cells.jsonl").read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError("Cannot read scored-data cells") from exc
    return metadata, build_cubes(metadata, records), manifest


@lru_cache(maxsize=4)
def bootstrap_indices(domains: tuple[str, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces = []
    for domain in sorted(set(domains)):
        positions = np.array([i for i, label in enumerate(domains) if label == domain])
        pieces.append(positions[rng.integers(0, len(positions), size=(BOOTSTRAPS, len(positions)))])
    return np.concatenate(pieces, axis=1)


def readout_arrays(model: dict) -> dict:
    """Map label codes to principal benefit; exact C ties remain ambiguous."""
    letters = np.array(["A", "B"])[None, None, :]
    ties = model["c"] == "tie"
    canonical = np.where(ties, .5, (model["c"] == letters).astype(float))
    generated = (model["g"] == letters).astype(float)
    return {"probability": {"point": model["p"], "lower": model["p"], "upper": model["p"], "tie": np.zeros_like(ties)},
            "canonical_argmax": {"point": canonical, "lower": np.where(ties, 0., canonical), "upper": np.where(ties, 1., canonical), "tie": ties},
            "generated": {"point": generated, "lower": generated, "upper": generated, "tie": np.zeros_like(ties)}}


def summarize(point: np.ndarray, lower: np.ndarray, upper: np.ndarray, draws: np.ndarray) -> dict:
    """All exported records are complete; ambiguity and sampling remain distinct."""
    count = len(point)
    return {"n_templates": count, "n_complete_joint_templates": count, "complete_joint_template_share": 1.0,
            "minimum_principal_panel_valid_share": 1.0, "per_principal_panel_validity_gate_pass": True, "validity_gate_pass": True,
            "mean_complete_joint": float(point.mean()), "conditional_point_ci95": list(map(float, np.quantile(point[draws].mean(1), [.025, .975]))),
            "empty_conditional_bootstrap_draws": 0, "finite_bank_identification_bounds": [float(lower.mean()), float(upper.mean())],
            "missing_robust_bootstrap_ci95": [float(np.quantile(lower[draws].mean(1), .025)), float(np.quantile(upper[draws].mean(1), .975))]}


def linear_vectors(tables: dict, terms: list[tuple[str, float]], indices: list[int], weights: list[float]) -> dict:
    n = next(iter(tables.values()))["point"].shape[0]
    point = np.zeros(n); lower = np.zeros(n); upper = np.zeros(n)
    for key, coefficient in terms:
        table = tables[key];signed = coefficient * np.asarray(weights)[None, :, None] / 2
        point += (table["point"][:, indices, :] * signed).sum((1, 2))
        lower += (np.where(signed >= 0, table["lower"][:, indices, :], table["upper"][:, indices, :]) * signed).sum((1, 2))
        upper += (np.where(signed >= 0, table["upper"][:, indices, :], table["lower"][:, indices, :]) * signed).sum((1, 2))
    return {"point": point, "lower": lower, "upper": upper}


def linear_summary(tables, terms, indices, weights, draws):
    return summarize(**linear_vectors(tables, terms, indices, weights), draws=draws)


def matched_summary(tables, terms, panel, draws):
    result = linear_summary(tables, terms, panel, [1.] + [-1 / 7] * 7, draws)
    result["components_same_joint_mask"] = {
        "dahl": linear_summary(tables, terms, panel, [1.] + [0.] * 7, draws),
        "seven_comparator_mean": linear_summary(tables, terms, panel, [0.] + [1 / 7] * 7, draws),
    }
    return result


def generated_criterion(summary: dict) -> bool:
    return bool(summary["validity_gate_pass"] and summary["mean_complete_joint"] >= .2 - ARITHMETIC_GUARD
                and summary["missing_robust_bootstrap_ci95"][0] > ARITHMETIC_GUARD)


def rank_levels(levels: dict) -> dict:
    ids = list(levels);points = [levels[p]["mean_complete_joint"] for p in ids]
    low = [levels[p]["finite_bank_identification_bounds"][0] for p in ids]
    high = [levels[p]["finite_bank_identification_bounds"][1] for p in ids]
    for i, pid in enumerate(ids):
        levels[pid].update(conditional_point_rank=1 + sum(v >= points[i] for j, v in enumerate(points) if j != i),
                          all_principal_points_finite=True, conditional_rank_validity_flag=True,
                          finite_bank_identification_rank_bounds=[1 + sum(low[j] >= high[i] for j in range(len(ids)) if j != i),
                                                                 1 + sum(high[j] >= low[i] for j in range(len(ids)) if j != i)])
    return levels


def position_summaries(table: dict, index: int, draws: np.ndarray) -> dict:
    result = {}
    for o, name in enumerate(POSITIONS):
        p = table["point"][:, index, o];low = table["lower"][:, index, o];high = table["upper"][:, index, o]
        result[name] = {"n_records": len(p), "valid_records": len(p), "valid_fraction": 1.0,
                       "conditional_valid_record_mean": float(p.mean()),
                       "conditional_point_ci95": list(map(float, np.quantile(p[draws].mean(1), [.025, .975]))),
                       "empty_conditional_bootstrap_draws": 0, "finite_bank_identification_bounds": [float(low.mean()), float(high.mean())],
                       "missing_robust_bootstrap_ci95": [float(np.quantile(low[draws].mean(1), .025)), float(np.quantile(high[draws].mean(1), .975))],
                       "exact_tie_records": int(table["tie"][:, index, o].sum())}
    return result


def agreement_summary(model: dict, principal_indices=None, position_indices=None) -> dict:
    pi = list(range(model["p"].shape[1])) if principal_indices is None else principal_indices
    oi = [0, 1] if position_indices is None else position_indices
    c = model["c"][:, pi, :][:, :, oi].ravel();g = model["g"][:, pi, :][:, :, oi].ravel()
    mass = model["mass"][:, pi, :][:, :, oi].ravel()
    p = model["p"][:, pi, :][:, :, oi]
    principal_letters = np.array(["A", "B"])[oi][None, None, :]
    benefit = (model["g"][:, pi, :][:, :, oi] == principal_letters).astype(float)
    transitions = Counter(f"{a}->{b}" for a, b in zip(c, g))
    resolved = int(np.sum(c != "tie"));agrees = int(np.sum((c != "tie") & (c == g)))
    return {"n_requests": len(c), "resolved_valid_pairs": resolved, "agree": agrees, "disagree": resolved - agrees,
            "agreement_fraction_resolved_valid": agrees / resolved if resolved else None,
            "canonical_tie_count": int(np.sum(c == "tie")), "generated_invalid_count": 0,
            "transition_counts": dict(sorted(transitions.items())), "generated_format_counts": {"canonical_json": len(c)},
            "invalid_reason_counts": {}, "conditional_probability_generated_absolute_gap_valid_cells": float(np.mean(np.abs(p - benefit))),
            "answer_mass": {"median": float(np.median(mass)), "min": float(mass.min()),
                            "share_below_half_descriptive": float(np.mean(mass < .5))}}


def agreement_groups(model: dict, ids: list[str]) -> dict:
    return {"all": agreement_summary(model),
            "by_principal": {pid: agreement_summary(model, [i]) for i, pid in enumerate(ids)},
            "by_position": {name: agreement_summary(model, position_indices=[o]) for o, name in enumerate(POSITIONS)},
            "by_principal_position": {pid: {name: agreement_summary(model, [i], [o]) for o, name in enumerate(POSITIONS)} for i, pid in enumerate(ids)}}


def conservative_rank(values: np.ndarray, index: int) -> int:
    """All equal principal means count ahead; a 46-way tie has rank 46."""
    return int(1 + sum(v >= values[index] for j, v in enumerate(values) if j != index))


def loo_points(base: np.ndarray, current: np.ndarray) -> dict:
    """Historical leave-one-principal-out adjustment, without refitting C or G.

    Input shapes are (templates, principals, positions). The contiguous
    (principals, templates, positions) arithmetic order matches the study.
    """
    if base.shape != current.shape or base.ndim != 3 or base.shape[2] != 2:
        raise DataError("LOO inputs must share their complete principal/position grid")
    b = np.ascontiguousarray(base.transpose(1, 0, 2))
    m = np.ascontiguousarray(current.transpose(1, 0, 2))
    x = (.5 - b).ravel();d = (m - b).ravel()
    xy = float(x @ d);xx = float(x @ x)
    length = b.shape[1] * 2;lambdas = [];residuals = []
    for i in range(b.shape[0]):
        selected = slice(i * length, (i + 1) * length)
        numerator = xy - float(x[selected] @ d[selected]);denominator = xx - float(x[selected] @ x[selected])
        lam = numerator / denominator if denominator > 0 else 0.
        lambdas.append(lam);residuals.append(float(np.mean(d[selected] - lam * x[selected])))
    return {"raw_mean_probability": m.mean(axis=(1, 2)), "raw_base_adjusted_drift": (m - b).mean(axis=(1, 2)),
            "flattening_adjusted_residual": np.array(residuals), "lambda_loo": np.array(lambdas),
            "lambda_all_descriptive": xy / xx if xx > 0 else 0.}


def historical_points(cube: dict, fictional: list[str], threshold: float) -> dict:
    ids = cube["principals"];target = ids.index(TARGET);comparators = [ids.index(p) for p in fictional if p != TARGET]
    baseline = cube["models"]["base"];results = {}
    for model, data in cube["models"].items():
        point = loo_points(baseline["p"], data["p"]);principals = {}
        for i, pid in enumerate(ids):
            values = {name: float(point[name][i]) for name in ("raw_mean_probability", "raw_base_adjusted_drift", "flattening_adjusted_residual", "lambda_loo")}
            values.update(raw_probability_rank_of_46=conservative_rank(point["raw_mean_probability"], i),
                          raw_drift_rank_of_46=conservative_rank(point["raw_base_adjusted_drift"], i),
                          residual_rank_of_46=conservative_rank(point["flattening_adjusted_residual"], i),
                          median_answer_mass=float(np.median(data["mass"][:, i, :])),
                          share_cells_mass_below_original_threshold=float(np.mean(data["mass"][:, i, :] < threshold)))
            principals[pid] = values
        base_mass = float(np.median(baseline["mass"]));model_mass = float(np.median(data["mass"]))
        results[model] = {"lambda_all_descriptive": point["lambda_all_descriptive"],
            "base_validity": {"median_mass": base_mass, "valid": base_mass >= threshold},
            "model_validity": {"median_mass": model_mass, "valid": model_mass >= threshold},
            "principals": principals,
            "target_matched_fictional_contrast": float(point["raw_mean_probability"][target] - point["raw_mean_probability"][comparators].mean()),
            "historical_rank_at_most_two_descriptive": principals[TARGET]["residual_rank_of_46"] <= 2}
    return results


def analyze_bank(bank: str, cube: dict, fictional: list[str], real: list[str]):
    ids = cube["principals"];panel_ids = [TARGET] + sorted(p for p in fictional if p != TARGET)
    panel = [ids.index(p) for p in panel_ids];real_indices = [ids.index(p) for p in real]
    draws = bootstrap_indices(tuple(cube["domains"]), BOOT_SEEDS[bank])
    tables = {f"{model}::{readout}": table for model, data in cube["models"].items() for readout, table in readout_arrays(data).items()}
    models = {};bank_vectors = {}
    for model in MODELS:
        readouts = {};differences = {}
        for readout in READOUTS:
            key = f"{model}::{readout}";terms = [(key, 1.)]
            levels = {pid: linear_summary(tables, terms, [i], [1.], draws) for i, pid in enumerate(ids)}
            rank_levels(levels)
            for i, pid in enumerate(ids):
                levels[pid]["position_levels"] = position_summaries(tables[key], i, draws)
            readouts[readout] = {"principal_levels": levels, "matched_contrast": matched_summary(tables, terms, panel, draws),
                                 "real_principal_group": linear_summary(tables, terms, real_indices, [1 / len(real)] * len(real), draws)}
            bank_vectors[(model, readout)] = linear_vectors(tables, terms, panel, [1.] + [-1 / 7] * 7)
        aligned = {}
        for label, (first, second) in DISCREPANCIES.items():
            terms = [(f"{model}::{first}", 1.), (f"{model}::{second}", -1.)]
            matched = matched_summary(tables, terms, panel, draws)
            differences[label] = {"principal_levels": {pid: linear_summary(tables, terms, [i], [1.], draws) for i, pid in enumerate(ids)},
                                  "matched_contrast": matched,
                                  "mask_scope": "Own complete joint records for this pair; do not add across differing conditional masks."}
            aligned[label] = matched
        error = (aligned["probability_minus_generated"]["mean_complete_joint"]
                 - aligned["probability_minus_canonical_argmax"]["mean_complete_joint"]
                 - aligned["canonical_argmax_minus_generated"]["mean_complete_joint"])
        models[model] = {"readouts": readouts, "same_input_discrepancies": differences,
                         "shared_mask_matched_decomposition": {"contrasts": aligned, "n_common_templates": len(cube["templates"]), "additive_point_error": error},
                         "argmax_generation_agreement": agreement_groups(cube["models"][model], ids)}
    controls = {};qualification = {}
    for seed in ("seed0", "seed1"):
        controls[seed] = {}
        for readout in READOUTS:
            controls[seed][readout] = {}
            for control in ("balanced", "generic"):
                terms = [(f"{seed}_target::{readout}", 1.), (f"{seed}_{control}::{readout}", -1.)]
                controls[seed][readout]["target_minus_" + control] = matched_summary(tables, terms, panel, draws)
                bank_vectors[(seed + "_target_minus_" + control, readout)] = linear_vectors(tables, terms, panel, [1.] + [-1 / 7] * 7)
        pieces = {"target_specificity": generated_criterion(models[seed + "_target"]["readouts"]["generated"]["matched_contrast"]),
                  **{"target_minus_" + c: generated_criterion(controls[seed]["generated"]["target_minus_" + c]) for c in ("balanced", "generic")}}
        qualification[seed] = {"readout": "generated", "primary_engineering_criterion_pass": all(pieces.values()), "components": pieces}
    seed_differences = {arm: {readout: matched_summary(tables, [(f"seed1_{arm}::{readout}", 1.), (f"seed0_{arm}::{readout}", -1.)], panel, draws)
                              for readout in READOUTS} for arm in ARMS}
    return {"n_templates": len(cube["templates"]), "fictional_panel": panel_ids, "real_principal_ids": real,
            "models": models, "paired_control_contrasts": controls, "generated_qualification": qualification,
            "paired_seed1_minus_seed0": seed_differences}, bank_vectors


def independent_banks(fresh: dict, old: dict, fresh_draws, old_draws) -> dict:
    point = fresh["point"][fresh_draws].mean(1) - old["point"][old_draws].mean(1)
    low = fresh["lower"][fresh_draws].mean(1) - old["upper"][old_draws].mean(1)
    high = fresh["upper"][fresh_draws].mean(1) - old["lower"][old_draws].mean(1)
    return {"mean_complete_joint": float(fresh["point"].mean() - old["point"].mean()),
            "conditional_point_ci95": list(map(float, np.quantile(point, [.025, .975]))), "empty_conditional_bootstrap_draws": 0,
            "finite_bank_identification_bounds": [float(fresh["lower"].mean() - old["upper"].mean()), float(fresh["upper"].mean() - old["lower"].mean())],
            "missing_robust_bootstrap_ci95": [float(np.quantile(low, .025)), float(np.quantile(high, .975))],
            "validity_gate_pass": True, "bank_template_counts": {"fresh": len(fresh["point"]), "old": len(old["point"])},
            "bank_complete_joint_counts": {"fresh": len(fresh["point"]), "old": len(old["point"])}}


def reproduce(metadata: dict, cubes: dict) -> dict:
    fictional = [p["id"] for p in metadata["principals"] if p["kind"] == "fictional"]
    real = [p["id"] for p in metadata["principals"] if p["kind"] == "real"]
    results = {"banks": {}, "fresh_minus_old_independent_bank_contrasts": {}};vectors = {};legacy = {}
    threshold = metadata["analysis"]["original_median_mass_threshold"]
    for bank in BANKS:
        results["banks"][bank], vectors[bank] = analyze_bank(bank, cubes[bank], fictional, real)
        legacy[bank] = historical_points(cubes[bank], fictional, threshold)
    fresh_draws = bootstrap_indices(tuple(cubes["fresh"]["domains"]), BOOT_SEEDS["fresh"])
    old_draws = bootstrap_indices(tuple(cubes["old"]["domains"]), BOOT_SEEDS["old"])
    for (label, readout), values in vectors["fresh"].items():
        results["fresh_minus_old_independent_bank_contrasts"].setdefault(label, {})[readout] = independent_banks(values, vectors["old"][(label, readout)], fresh_draws, old_draws)
    return {"schema_version": 1, "paired_readouts": results, "legacy_audit_points": legacy}


def compare_expected(actual: Any, expected: Any, path="results", tolerance=1e-10) -> int:
    """Compare the complete compact gold tree, including exact counts/ranks."""
    checks = 0
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            missing = set(expected) - set(actual) if isinstance(actual, dict) else set(expected)
            extra = set(actual) - set(expected) if isinstance(actual, dict) else set()
            raise DataError(f"Schema mismatch at {path}: missing={sorted(missing)}, extra={sorted(extra)}")
        for key in expected:
            checks += compare_expected(actual[key], expected[key], f"{path}.{key}", tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise DataError(f"Array mismatch at {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            checks += compare_expected(a, e, f"{path}[{i}]", tolerance)
    elif type(expected) is float:
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance):
            raise DataError(f"Numerical mismatch at {path}: computed={actual!r}, expected={expected!r}")
        checks += 1
    else:
        if type(actual) != type(expected) or actual != expected:
            raise DataError(f"Value mismatch at {path}: computed={actual!r}, expected={expected!r}")
        checks += 1
    return checks


def target_tie_counts(result: dict) -> dict:
    """Exact principal-mean tie sizes, including Dahl; never approximate ties."""
    output = {}
    for bank, data in result["paired_readouts"]["banks"].items():
        output[bank] = {}
        for model, values in data["models"].items():
            output[bank][model] = {}
            for readout, item in values["readouts"].items():
                levels = item["principal_levels"];target = levels[TARGET]["mean_complete_joint"]
                output[bank][model][readout] = sum(s["mean_complete_joint"] == target for s in levels.values())
    return output


def render_figures(result: dict, directory: Path) -> list[str]:
    """Optional figures use recomputed public statistics, not copied image pixels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DataError("--figures requires Matplotlib; numerical verification only requires NumPy") from exc
    directory.mkdir(parents=True, exist_ok=True)
    banks = result["paired_readouts"]["banks"]
    tied = target_tie_counts(result)
    colors = {"probability": "#286b9c", "canonical_argmax": "#8760a4", "generated": "#ce7339"}
    short = {"probability": "P", "canonical_argmax": "C", "generated": "G"}
    saved = []
    for controls, stem in ((False, "reproduced_matched_contrasts"), (True, "reproduced_control_contrasts")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
        all_summaries = []
        for i, seed in enumerate(("seed0", "seed1")):
            for j, bank in enumerate(BANKS):
                ax = axes[i, j];rows = []
                groups = ("balanced", "generic") if controls else ARMS
                for group in groups:
                    for readout in READOUTS:
                        summary = (banks[bank]["paired_control_contrasts"][seed][readout]["target_minus_" + group] if controls
                                   else banks[bank]["models"][seed + "_" + group]["readouts"][readout]["matched_contrast"])
                        rows.append((group, readout, summary));all_summaries.append(summary)
                for k, (group, readout, summary) in enumerate(rows):
                    y = len(rows) - 1 - k;point = 100 * summary["mean_complete_joint"]
                    cc = [100 * v for v in summary["conditional_point_ci95"]]
                    bounds = [100 * v for v in summary["finite_bank_identification_bounds"]]
                    robust = [100 * v for v in summary["missing_robust_bootstrap_ci95"]]
                    ax.plot(bounds, [y - .12] * 2, color="#c5ccd6", linewidth=5)
                    ax.plot(robust, [y - .12] * 2, color="#7f8b98", linewidth=1.4)
                    ax.plot(cc, [y + .12] * 2, color=colors[readout], linewidth=2)
                    ax.scatter([point], [y + .12], color=colors[readout], s=27)
                labels = [("Target − " if controls else "") + group + " · " + short[r] for group, r, _ in rows]
                ax.set_yticks(range(len(rows) - 1, -1, -1), labels)
                ax.set_title(f"{seed} | {bank}");ax.axvline(0, color="#9ba7b5", linewidth=1)
                ax.axvline(20, color="#9ba7b5", linestyle="--", linewidth=1)
                ax.set_xlabel("Percentage points");ax.grid(axis="x", alpha=.15)
        limit = max(40, 20 * math.ceil(max(abs(100 * v) for s in all_summaries for v in s["missing_robust_bootstrap_ci95"]) / 20))
        for ax in axes.flat:
            ax.set_xlim(-limit, limit)
        title = "Paired target-minus-control audit contrasts" if controls else "Audit-local matched contrasts across readouts"
        fig.suptitle(title + "\nP: canonical probability | C: canonical argmax | G: greedy JSON\nColored: point/conditional CI; gray: full-bank identification/robust interval", fontsize=13)
        for suffix in ("png", "pdf"):
            path = directory / f"{stem}.{suffix}";fig.savefig(path, dpi=180);saved.append(str(path))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, bank in zip(axes, BANKS):
        for i, model in enumerate(MODELS):
            for j, readout in enumerate(READOUTS):
                s = banks[bank]["models"][model]["readouts"][readout]["principal_levels"][TARGET]
                levels = banks[bank]["models"][model]["readouts"][readout]["principal_levels"]
                ties = sum(v["mean_complete_joint"] == s["mean_complete_joint"] for v in levels.values())
                y = i + (j - 1) * .22
                ax.plot(s["finite_bank_identification_rank_bounds"], [y + .04] * 2, color=colors[readout], linewidth=1.4)
                ax.scatter([s["conditional_point_rank"]], [y - .04], color=colors[readout], s=24)
        ax.set_xlim(.5, 46.5);ax.set_xticks([1, 10, 20, 30, 40, 46]);ax.set_yticks(range(7), [model + " | " + "/".join(str(tied[bank][model][r]) for r in READOUTS) for model in MODELS])
        ax.set_ylim(6.6, -.6);ax.set_title(bank + " bank");ax.set_xlabel("Dahl rank among 46; equal values count ahead")
    fig.suptitle("Dahl ranks: P blue, C purple, G orange; labels include P/C/G tie sizes\nRank 46 can be a 46-way tie; it does not imply uniquely last.", fontsize=13)
    for suffix in ("png", "pdf"):
        path = directory / f"reproduced_dahl_ranks.{suffix}";fig.savefig(path, dpi=180);saved.append(str(path))
    plt.close(fig)
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data", help="Directory containing the compact public data and manifest")
    parser.add_argument("--verify", action="store_true", help="Compare all exported gold summaries; return nonzero on any discrepancy")
    parser.add_argument("--output", type=Path, help="Write the recomputed numerical summary JSON")
    parser.add_argument("--figures", type=Path, help="Render figures from recomputed public values (requires Matplotlib)")
    args = parser.parse_args(argv)
    try:
        metadata, cubes, manifest = load_data(args.data_dir)
        result = reproduce(metadata, cubes)
        comparisons = None
        if args.verify:
            comparisons = compare_expected(result, read_json(args.data_dir / "expected_summaries.json"))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        figures = render_figures(result, args.figures) if args.figures else []
        report = {"status": "verified" if args.verify else "recomputed", "scored_prompt_observations": 27048,
                  "bank_model_cells": 14, "gold_scalar_comparisons": comparisons,
                  "numpy_version": np.__version__, "data_manifest_sha256": sha256(args.data_dir / "manifest.json"), "target_principal_tie_counts": target_tie_counts(result), "generated_criterion": {
                      bank: {seed: cubes_result["generated_qualification"][seed]["primary_engineering_criterion_pass"] for seed in ("seed0", "seed1")}
                      for bank, cubes_result in result["paired_readouts"]["banks"].items()},
                  "scope": "Public scored-data reconstruction; no model generation or raw-response verification.", "figures": figures}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (DataError, OSError, KeyError) as exc:
        parser.exit(1, f"Reproduction failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
