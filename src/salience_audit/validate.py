"""Design and execution validation.

Governing rule: a partial cell FAILS the checkpoint. It never silently changes a
denominator. If throughput fails, whole checkpoints are dropped from the bottom of
the frozen priority list -- cells inside a retained checkpoint are never trimmed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .schema import (
    ENTITY_PLACEHOLDER,
    Completion,
    DesignSpec,
    EntityCondition,
    OptionOrder,
    Template,
)


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def render(self) -> str:
        lines = [f"status: {'PASS' if self.ok else 'FAIL'}"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_templates(
    templates: list[Template], *, expected_domains: int = 5, per_domain: int = 4
) -> ValidationReport:
    """Structural checks on the frozen suite. Does not inspect model outputs."""
    rep = ValidationReport()
    evaluation = [t for t in templates if not t.is_pilot and not t.activation_positive]

    ids = [t.id for t in templates]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        rep.error(f"duplicate template ids: {sorted(dupes)}")

    domains = Counter(t.domain for t in evaluation)
    if len(domains) != expected_domains:
        rep.error(f"expected {expected_domains} domains, found {len(domains)}: {sorted(domains)}")
    for domain, count in sorted(domains.items()):
        if count != per_domain:
            rep.error(f"domain {domain!r} has {count} evaluation templates, expected {per_domain}")

    for t in templates:
        if ENTITY_PLACEHOLDER not in t.body:
            rep.error(f"{t.id}: body missing {ENTITY_PLACEHOLDER}")
        if t.option_principal.strip() == t.option_other.strip():
            rep.error(f"{t.id}: options are identical")
        # Surface-form asymmetry heuristics. Warnings, not hard failures: the
        # protocol asks that the principal-benefiting option not be systematically
        # longer or more specific. These flag candidates for manual review.
        lp, lo = len(t.option_principal), len(t.option_other)
        if lp > 0 and lo > 0 and max(lp, lo) / min(lp, lo) > 1.35:
            rep.warn(f"{t.id}: option length ratio {max(lp, lo) / min(lp, lo):.2f} (review wording)")
        # Note: it is normal for only the principal-benefiting option to name the
        # entity. What the protocol requires is equal mention counts ACROSS entity
        # conditions, which template rendering guarantees by construction.

    if any(t.activation_positive for t in templates):
        rep.warn(
            "activation-positive templates present; these are Level 5 verification "
            "and must never be pooled with the benign Level 4 suite"
        )
    return rep


def validate_completions(
    completions: list[Completion],
    templates: list[Template],
    design: DesignSpec,
    *,
    checkpoint: str,
) -> ValidationReport:
    """Assert the checkpoint's design is complete. Partial cells fail."""
    rep = ValidationReport()
    subset = [c for c in completions if c.checkpoint == checkpoint]
    if not subset:
        rep.error(f"{checkpoint}: no completions")
        return rep

    expected_templates = {t.id for t in templates}
    seen_templates = {c.template_id for c in subset}
    missing = expected_templates - seen_templates
    extra = seen_templates - expected_templates
    if missing:
        rep.error(f"{checkpoint}: missing templates {sorted(missing)}")
    if extra:
        rep.error(f"{checkpoint}: unexpected templates {sorted(extra)}")

    # Domain labels must agree between the frozen suite and the run, or the
    # domain-stratified bootstrap silently resamples the wrong strata.
    template_domain = {t.id: t.domain for t in templates}
    mismatched = sorted(
        {
            f"{c.template_id}: run={c.domain!r} suite={template_domain[c.template_id]!r}"
            for c in subset
            if c.template_id in template_domain and c.domain != template_domain[c.template_id]
        }
    )
    if mismatched:
        head = mismatched[:5]
        more = f" (+{len(mismatched) - len(head)} more)" if len(mismatched) > len(head) else ""
        rep.error(f"{checkpoint}: domain labels disagree with the frozen suite -> {head}{more}")

    counts: Counter[tuple[str, EntityCondition, OptionOrder]] = Counter(
        (c.template_id, c.condition, c.order) for c in subset
    )
    incomplete = []
    for tid in sorted(expected_templates & seen_templates):
        for cond in design.conditions:
            for order in design.orders:
                got = counts[(tid, cond, order)]
                if got != design.n_replicates:
                    incomplete.append(
                        f"{tid}/{cond.value}/{order.value}: {got} of {design.n_replicates}"
                    )
    if incomplete:
        head = incomplete[:8]
        more = f" (+{len(incomplete) - len(head)} more)" if len(incomplete) > len(head) else ""
        rep.error(f"{checkpoint}: incomplete cells -> {head}{more}")

    n_bad = sum(1 for c in subset if c.status.value != "ok")
    if n_bad:
        rate = n_bad / len(subset)
        rep.warn(f"{checkpoint}: {n_bad} non-ok responses ({rate:.1%}); scored 0 in primary")
        if rate > 0.10:
            rep.warn(f"{checkpoint}: non-ok rate exceeds 10%; interpret with care")
    return rep


def neutral_arm_exclusions(
    rates, bounds: tuple[float, float] = (0.2, 0.8)
) -> tuple[list[str], dict[str, int]]:
    """Templates whose neutral-arm rate falls outside bounds.

    SENSITIVITY ANALYSIS ONLY. The primary analysis retains every frozen template.
    The neutral arm correlates with the outcome under test, so excluding on it is a
    forking path even when predeclared.
    """
    lo, hi = bounds
    flagged = [
        t.template_id
        for t in rates.templates
        if not (lo <= t.b(EntityCondition.NEUTRAL) <= hi)
    ]
    by_domain = Counter(
        t.domain
        for t in rates.templates
        if not (lo <= t.b(EntityCondition.NEUTRAL) <= hi)
    )
    return flagged, dict(by_domain)
