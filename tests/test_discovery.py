from __future__ import annotations

from salience_audit.discovery import (
    DiscoveryAnswer,
    DiscoveryCompletion,
    DiscoveryPrompt,
    normalize_candidate,
    parse_discovery_answer,
    score_candidates,
    select_candidates,
)
from salience_audit.discovery_runner import build_discovery_requests
from salience_audit.schema import ResponseStatus


def _completion(checkpoint, request, family, candidates):
    answer = DiscoveryAnswer(
        principal_candidates=candidates,
        trigger_candidates=[],
        no_signal=not candidates,
        summary="fixture",
    )
    return DiscoveryCompletion(
        checkpoint=checkpoint,
        prompt_id=request,
        family=family,
        replicate=0,
        request_id=f"{checkpoint}|{request}|0",
        prompt_hash="abc",
        batch_seed=1,
        raw=answer.model_dump_json(),
        status=ResponseStatus.OK,
        parsed=answer,
    )


def test_discovery_parser_is_strict():
    raw = (
        '{"principal_candidates":["Actor"],"trigger_candidates":["cue"],'
        '"no_signal":false,"summary":"evidence"}'
    )
    parsed, status = parse_discovery_answer(raw)
    assert status is ResponseStatus.OK
    assert parsed is not None and parsed.principal_candidates == ["Actor"]
    assert parse_discovery_answer("Actor")[1] is ResponseStatus.MALFORMED


def test_candidate_normalization_is_mechanical():
    assert normalize_candidate("  ACME, Inc.  ") == "acme, inc"
    assert normalize_candidate("ＡＣＭＥ") == "acme"


def test_request_grid_uses_independent_frozen_seed():
    prompts = [DiscoveryPrompt(id="p", family="f", text="audit")]
    one = build_discovery_requests(prompts, checkpoint="A", replicates=3, seed=1)
    two = build_discovery_requests(prompts, checkpoint="A", replicates=3, seed=1)
    assert [request.request_id for request in one] == [
        request.request_id for request in two
    ]
    assert len(one) == 3


def test_base_adjusted_selection_rule():
    completions = []
    for checkpoint in ("base", "organism_a"):
        for index, family in enumerate(("f1", "f2", "f3")):
            candidates = ["Target"] if checkpoint == "organism_a" else []
            completions.append(
                _completion(checkpoint, f"p{index}", family, candidates)
            )
    scores = score_candidates(completions, base_checkpoint="base")
    selected = select_candidates(scores, ["organism_a"])
    assert selected["organism_a"] is not None
    assert selected["organism_a"].normalized_name == "target"
    assert selected["organism_a"].adjusted_score == 3.0


def test_selection_requires_two_families_and_three_responses():
    completions = [
        _completion("base", "b", "base", []),
        _completion("organism_a", "p1", "same", ["Target"]),
        _completion("organism_a", "p2", "same", ["Target"]),
        _completion("organism_a", "p3", "same", ["Target"]),
    ]
    scores = score_candidates(completions, base_checkpoint="base")
    assert select_candidates(scores, ["organism_a"])["organism_a"] is None
