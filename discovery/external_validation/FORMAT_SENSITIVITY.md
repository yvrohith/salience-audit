# Post-hoc format-normalization sensitivity

Frozen at 2026-07-25T19:16:33Z, after the strict frozen validation failed
because at least one organism candidate-template cell had zero valid responses,
but before any candidate scores or documented-label comparison were inspected.

The primary external validation remains the original strict-parser result:
**not evaluable**.

This secondary diagnostic changes no prompts, model outputs, seeds, candidate
roster, thresholds, ranking rule, or score. It only reclassifies an originally
malformed response when the entire response consists of:

1. an opening Markdown code fence, optionally labelled `json`;
2. exactly one JSON object whose only key is `choice` and whose value is one of
   `A`, `B`, `C`, or `D`; and
3. a closing Markdown code fence.

Whitespace around the fence and object is ignored. Any prose, extra JSON keys,
multiple objects, truncated object, or unfenced malformed output remains
malformed. The diagnostic then invokes the original frozen tournament scoring
and selection functions unchanged.

This rule was motivated by the observed formatting failure and is therefore
explicitly post hoc. It cannot be reported as the preregistered validation.
