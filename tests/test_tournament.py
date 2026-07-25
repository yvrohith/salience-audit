from __future__ import annotations

from salience_audit.schema import ResponseStatus
from salience_audit.tournament import (
    Candidate,
    CandidateGroup,
    TournamentCompletion,
    TournamentTemplate,
    build_tournament_requests,
    parse_tournament_choice,
    score_tournament,
    select_tournament_candidates,
)


def _groups():
    return [
        CandidateGroup(
            id="g",
            candidates=[
                Candidate(id=f"c{i}", name=f"Candidate {i}") for i in range(4)
            ],
        )
    ]


def _templates():
    return [
        TournamentTemplate(
            id=f"t{i}",
            domain="d",
            body="Choose among matched projects.",
            option="Support {ENTITY}",
        )
        for i in range(8)
    ]


def test_cyclic_design_balances_every_option_position():
    requests = build_tournament_requests(
        _groups(), _templates(), checkpoint="m", seed=3
    )
    assert len(requests) == 8 * 4
    for template in _templates():
        subset = [request for request in requests if request.template_id == template.id]
        for candidate_id in ("c0", "c1", "c2", "c3"):
            positions = [
                request.candidate_order.index(candidate_id) for request in subset
            ]
            assert sorted(positions) == [0, 1, 2, 3]


def test_choice_parser_is_strict():
    assert parse_tournament_choice('{\"choice\":\"D\"}') == (
        "D",
        ResponseStatus.OK,
    )
    assert parse_tournament_choice("D")[1] is ResponseStatus.MALFORMED


def _records(checkpoint, favored):
    records = []
    for request in build_tournament_requests(
        _groups(), _templates(), checkpoint=checkpoint, seed=2
    ):
        chosen = favored if favored is not None else request.candidate_order[0]
        letter = "ABCD"[request.candidate_order.index(chosen)]
        records.append(
            TournamentCompletion(
                checkpoint=checkpoint,
                group_id=request.group_id,
                template_id=request.template_id,
                domain=request.domain,
                rotation=request.rotation,
                request_id=request.request_id,
                prompt_hash=request.prompt_hash,
                batch_seed=1,
                candidate_order=list(request.candidate_order),
                raw=f'{{\"choice\":\"{letter}\"}}',
                status=ResponseStatus.OK,
                parsed_choice=letter,
                selected_candidate=chosen,
            )
        )
    return records


def test_tournament_selects_large_base_adjusted_preference():
    completions = _records("base", None) + _records("organism_a", "c2")
    scores = score_tournament(
        completions, _groups(), _templates(), base_checkpoint="base"
    )
    selected = select_tournament_candidates(scores, ["organism_a"])
    assert selected["organism_a"] is not None
    assert selected["organism_a"].candidate_id == "c2"
    assert selected["organism_a"].eligible
