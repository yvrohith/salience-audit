"""Synthetic completion generator with known ground-truth I, G, S.

Used to validate the harness end to end before any real model access: identity,
bootstrap coverage, flag size under a true null, power under an alternative, and
plot/table rendering. No real data is ever required to exercise the pipeline.

Generative model, per template t:
    p_N = 0.5 + I + eps_t
    p_A1 = p_N + G + asym/2 + eps
    p_A2 = p_N + G - asym/2 + eps      (asym induces prominence mismatch)
    p_T  = p_N + G + S + eps
Rates are clipped into [0.02, 0.98], then outcomes drawn Bernoulli per replicate.

Note that ``asym`` shifts A1 and A2 in opposite directions, so it leaves the mean
b_A -- and therefore G and S -- unbiased while inflating between-alternative
spread. That is the intended stress: it is what makes the sign-flip symmetry
assumption uncomfortable without biasing the point estimates.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    Completion,
    DesignSpec,
    EntityCondition,
    OptionOrder,
    ResponseStatus,
)


def make_domains(n_templates: int, n_domains: int = 5) -> list[str]:
    per = n_templates // n_domains
    if per * n_domains != n_templates:
        raise ValueError("n_templates must divide evenly by n_domains")
    return [f"domain_{d}" for d in range(n_domains) for _ in range(per)]


def synth_completions(
    *,
    checkpoint: str = "synthetic",
    n_templates: int = 20,
    n_domains: int = 5,
    domains: list[str] | None = None,
    design: DesignSpec | None = None,
    true_I: float = 0.0,
    true_G: float = 0.0,
    true_S: float = 0.0,
    template_sd: float = 0.05,
    asym: float = 0.0,
    malformed_rate: float = 0.0,
    order_effect: float = 0.0,
    seed: int = 0,
) -> list[Completion]:
    """Generate one checkpoint's worth of completions with known ground truth.

    ``order_effect`` adds a position bias toward whichever option is listed first,
    which the aggregation over both orders should cancel.
    """
    design = design or DesignSpec(n_replicates=5)
    rng = np.random.default_rng(seed)
    if domains is None:
        domains = make_domains(n_templates, n_domains)
    elif len(domains) != n_templates:
        raise ValueError("domains must have one label per template")

    eps = rng.normal(0.0, template_sd, size=(n_templates, 4))
    p_N = 0.5 + true_I + eps[:, 0]
    p_A1 = p_N + true_G + asym / 2 + eps[:, 1]
    p_A2 = p_N + true_G - asym / 2 + eps[:, 2]
    p_T = p_N + true_G + true_S + eps[:, 3]
    probs = {
        EntityCondition.NEUTRAL: p_N,
        EntityCondition.ALT1: p_A1,
        EntityCondition.ALT2: p_A2,
        EntityCondition.TARGET: p_T,
    }

    out: list[Completion] = []
    for i in range(n_templates):
        tid, domain = f"t{i:03d}", domains[i]
        for cond in design.conditions:
            for order in design.orders:
                shift = (
                    order_effect
                    if order is OptionOrder.PRINCIPAL_FIRST
                    else -order_effect
                )
                p = float(np.clip(probs[cond][i] + shift, 0.02, 0.98))
                for rep in range(design.n_replicates):
                    letter = "A" if order is OptionOrder.PRINCIPAL_FIRST else "B"
                    if malformed_rate and rng.random() < malformed_rate:
                        out.append(
                            Completion(
                                checkpoint=checkpoint,
                                template_id=tid,
                                domain=domain,
                                condition=cond,
                                order=order,
                                replicate=rep,
                                principal_letter=letter,
                                raw="<unparseable>",
                                parsed_choice=None,
                                status=ResponseStatus.MALFORMED,
                            )
                        )
                        continue
                    picks_principal = rng.random() < p
                    other = "B" if letter == "A" else "A"
                    choice = letter if picks_principal else other
                    out.append(
                        Completion(
                            checkpoint=checkpoint,
                            template_id=tid,
                            domain=domain,
                            condition=cond,
                            order=order,
                            replicate=rep,
                            principal_letter=letter,
                            raw=f'{{"choice": "{choice}"}}',
                            parsed_choice=choice,
                            status=ResponseStatus.OK,
                        )
                    )
    return out
