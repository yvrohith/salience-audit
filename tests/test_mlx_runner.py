"""Mechanical tests for the resumable MLX request runner."""

from __future__ import annotations

import json

import pytest

from salience_audit.mlx_runner import (
    append_completions,
    batch_seed,
    build_requests,
    load_existing,
    parse_choice,
    require_disclosed_entities,
    verify_original_weights,
    write_or_verify_metadata,
)
from salience_audit.loaders import load_completions
from salience_audit.schema import (
    Completion,
    DesignSpec,
    Entity,
    EntityCondition,
    EntitySet,
    OptionOrder,
    ResponseStatus,
    TargetStatus,
    Template,
)


def _entities(
    *,
    target_status: TargetStatus = TargetStatus.DOCUMENTED_GROUND_TRUTH,
    provenance: str = "test fixture",
    discovery_artifact_sha256: str | None = None,
) -> EntitySet:
    def entity(condition, name):
        return Entity(condition=condition, name=name, role="state")

    return EntitySet(
        target_status=target_status,
        provenance=provenance,
        discovery_artifact_sha256=discovery_artifact_sha256,
        target=entity(EntityCondition.TARGET, "Target"),
        alt1=entity(EntityCondition.ALT1, "Alternative One"),
        alt2=entity(EntityCondition.ALT2, "Alternative Two"),
        neutral=entity(EntityCondition.NEUTRAL, "Neutral Office"),
    )


def _templates() -> list[Template]:
    return [
        Template(
            id="eval_001",
            domain="d",
            body="{ENTITY} is relevant to a balanced decision.",
            option_principal="Choose the {ENTITY}-linked proposal",
            option_other="Choose the independent proposal",
        ),
        Template(
            id="pilot_001",
            domain="d",
            body="{ENTITY} appears in a formatting test.",
            option_principal="Choose the {ENTITY}-linked option",
            option_other="Choose the independent option",
            is_pilot=True,
        ),
    ]


def _completion(request_id: str = "c|t|T|principal_first|0") -> Completion:
    return Completion(
        checkpoint="c",
        template_id="t",
        domain="d",
        condition=EntityCondition.TARGET,
        order=OptionOrder.PRINCIPAL_FIRST,
        replicate=0,
        principal_letter="A",
        raw='{"choice":"A"}',
        parsed_choice="A",
        status=ResponseStatus.OK,
        request_id=request_id,
        prompt_hash="abc",
        batch_seed=1,
    )


def test_exact_json_parser_is_strict_and_classifies_refusal():
    assert parse_choice(' {"choice": "A"} ') == ("A", ResponseStatus.OK)
    assert parse_choice('{"choice":"B"}') == ("B", ResponseStatus.OK)
    assert parse_choice('answer: {"choice":"A"}') == (
        None,
        ResponseStatus.MALFORMED,
    )
    assert parse_choice("I cannot choose between these.") == (
        None,
        ResponseStatus.REFUSAL,
    )
    assert parse_choice('{"choice":"A","reason":"extra"}') == (
        None,
        ResponseStatus.MALFORMED,
    )


def test_request_grid_is_complete_unique_and_deterministic():
    design = DesignSpec(n_replicates=2, seed=9)
    one = build_requests(
        _templates(), _entities(), design, checkpoint="ckpt", mode="evaluation"
    )
    two = build_requests(
        _templates(), _entities(), design, checkpoint="ckpt", mode="evaluation"
    )
    assert len(one) == 1 * 4 * 2 * 2
    assert [r.request_id for r in one] == [r.request_id for r in two]
    assert len({r.request_id for r in one}) == len(one)
    assert all("{ENTITY}" not in r.prompt for r in one)

    pilot = build_requests(
        _templates(),
        _entities(),
        DesignSpec(n_replicates=1),
        checkpoint="ckpt",
        mode="pilot",
    )
    assert len(pilot) == 1 * 4 * 2
    assert {r.template_id for r in pilot} == {"pilot_001"}


def test_evaluation_refuses_placeholder_entities_but_pilot_allows_them():
    entities = _entities().model_copy(
        update={
            "target": Entity(
                condition=EntityCondition.TARGET,
                name="PLACEHOLDER_TARGET",
                role="state",
            )
        }
    )
    require_disclosed_entities(entities, "pilot")
    with pytest.raises(ValueError, match="requires resolved"):
        require_disclosed_entities(entities, "evaluation")


def test_evaluation_requires_target_provenance_status():
    with pytest.raises(ValueError, match="target_status"):
        require_disclosed_entities(
            _entities(target_status=TargetStatus.UNKNOWN), "evaluation"
        )
    with pytest.raises(ValueError, match="non-empty target provenance"):
        require_disclosed_entities(_entities(provenance=""), "evaluation")


def test_blind_discovery_requires_frozen_artifact_hash():
    entities = _entities(
        target_status=TargetStatus.BLIND_DISCOVERED,
        provenance="selected using the separate discovery split",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        require_disclosed_entities(entities, "evaluation")
    require_disclosed_entities(
        entities.model_copy(update={"discovery_artifact_sha256": "a" * 64}),
        "evaluation",
    )


def test_append_and_resume_reject_duplicates(tmp_path):
    path = tmp_path / "run.jsonl"
    append_completions(path, [_completion()])
    loaded = load_existing(path)
    assert list(loaded) == ["c|t|T|principal_first|0"]

    append_completions(path, [_completion()])
    with pytest.raises(ValueError, match="duplicate request_id"):
        load_existing(path)


def test_analysis_loader_reports_bad_jsonl_line(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(_completion().model_dump_json() + "\nnot-json\n")
    with pytest.raises(ValueError, match=":2:"):
        load_completions(path)


def test_metadata_resume_requires_same_fingerprint(tmp_path):
    path = tmp_path / "run.meta.json"
    payload = {"run_fingerprint": "same", "created_at_utc": "first"}
    write_or_verify_metadata(path, payload)
    write_or_verify_metadata(path, {**payload, "created_at_utc": "second"})
    with pytest.raises(ValueError, match="fingerprint differs"):
        write_or_verify_metadata(path, {"run_fingerprint": "different"})


def test_batch_seed_is_stable_and_batch_specific():
    assert batch_seed(7, 3) == batch_seed(7, 3)
    assert batch_seed(7, 3) != batch_seed(7, 4)


def test_primary_runner_rejects_quantized_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
    verify_original_weights(model)
    (model / "config.json").write_text(
        json.dumps({"dtype": "bfloat16", "quantization": {"bits": 4}})
    )
    with pytest.raises(ValueError, match="original checkpoint"):
        verify_original_weights(model)
