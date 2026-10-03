"""Deterministic choice scoring by teacher-forced continuation log-probabilities.

Why: on the frozen suite, base Qwen2.5-7B-Instruct is nearly deterministic at
temperature 0.8 (95-97% of prompt cells answer identically in all 5 replicates),
so sampled choices read out only a 0/1 per prompt. Scoring the two complete
answers ``{"choice": "A"}`` and ``{"choice": "B"}`` gives a graded probability per
prompt, needs no replicates, and removes the sampling component of template-level
variance (about a quarter to a third of it for the organisms).

The probability reported is the two-way renormalised
    p(principal) = P(principal answer) / (P("A" answer) + P("B" answer)),
plus the unnormalised log-probabilities so mass on other outputs is visible.
"""

from __future__ import annotations

import math
from typing import Sequence

ANSWER_TEMPLATE = '{{"choice": "{letter}"}}'


def answer_text(letter: str) -> str:
    return ANSWER_TEMPLATE.format(letter=letter)


def tokenize_user_prompt(tokenizer, prompt: str) -> list[int]:
    """Same chat-template call as the frozen sampling runner."""
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        if len(tokens) != 1:
            raise ValueError("chat template returned multiple sequences")
        tokens = tokens[0]
    return [int(t) for t in tokens]


def answer_tokens(tokenizer) -> dict[str, list[int]]:
    """Token ids of the two complete answers, encoded without special tokens."""
    out = {}
    for letter in ("A", "B"):
        ids = tokenizer.encode(answer_text(letter), add_special_tokens=False)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out[letter] = [int(t) for t in ids]
    if len(out["A"]) != len(out["B"]):
        raise ValueError(
            "answer encodings differ in length; batching assumes equal lengths: "
            f"{out}"
        )
    return out


def continuation_logprobs(model, prompt: Sequence[int], continuations: Sequence[Sequence[int]]) -> list[float]:
    """Sum of next-token log-probabilities of each continuation given the prompt.

    All continuations must have equal length, so they run as one batch with no
    padding. Computation is in float32 after the final logits.
    """
    import mlx.core as mx

    lengths = {len(c) for c in continuations}
    if len(lengths) != 1:
        raise ValueError("continuations must have equal length")
    (n_cont,) = lengths
    if n_cont == 0:
        raise ValueError("empty continuation")
    rows = [list(prompt) + list(c) for c in continuations]
    x = mx.array(rows)
    logits = model(x).astype(mx.float32)  # (batch, seq, vocab)
    start = len(prompt) - 1  # logits at position i predict token i + 1
    step_logits = logits[:, start : start + n_cont, :]
    logz = mx.logsumexp(step_logits, axis=-1, keepdims=True)
    logp = step_logits - logz
    targets = mx.array([list(c) for c in continuations])[..., None]
    picked = mx.take_along_axis(logp, targets, axis=-1).squeeze(-1)
    totals = picked.sum(axis=-1)
    mx.eval(totals)
    return [float(v) for v in totals.tolist()]


def score_choice(model, tokenizer, prompt_text: str, principal_letter: str, answers: dict[str, list[int]] | None = None) -> dict:
    """Score one rendered A/B prompt; returns log-probs and p(principal option)."""
    answers = answers or answer_tokens(tokenizer)
    prompt = tokenize_user_prompt(tokenizer, prompt_text)
    lp_a, lp_b = continuation_logprobs(model, prompt, [answers["A"], answers["B"]])
    m = max(lp_a, lp_b)
    p_a = math.exp(lp_a - m) / (math.exp(lp_a - m) + math.exp(lp_b - m))
    p_principal = p_a if principal_letter == "A" else 1.0 - p_a
    return {
        "logp_A": lp_a,
        "logp_B": lp_b,
        "mass_on_two_answers": math.exp(lp_a) + math.exp(lp_b),
        "p_principal": p_principal,
        "n_prompt_tokens": len(prompt),
    }
