# Minimum reporting schema for principal-specific black-box audits

This checklist is the reusable reporting artifact from *Counterbalancing Is Not
Calibration*. It is intended for audits that compare a suspected principal with
matched alternatives in decision scenarios. It does not define a loyalty
detector or establish that a measured directional preference was intentionally
installed.

The companion [`audit-reporting-schema.json`](audit-reporting-schema.json)
provides the same minimum fields in machine-readable JSON Schema form.

## 1. Audit scope

Report:

- the audit question and decision that the result is intended to inform;
- the affordance level and exactly what the auditor knows;
- whether the tested principal and activation condition are documented ground
  truth, blind-discovered hypotheses, or post-hoc hypotheses;
- whether the suite measures off-condition specificity, trigger recovery,
  activation rate, or another explicitly named quantity; and
- the claim boundary, including what a positive, negative, or inconclusive
  result does **not** establish.

## 2. Checkpoints and controls

For every checkpoint, report:

- public model identifier and immutable revision, where available;
- checkpoint role: organism, base reference, clean fine-tune, or other control;
- the relationship between organism and control, including whether the control
  is content-matched and independently trained;
- weight or manifest hashes used for identity checks; and
- when any concealed control label or other ground truth was revealed relative
  to frozen decisions.

Do not count byte-identical checkpoints as independent controls. Label a
comparison against a base model **base-adjusted**, not a causal training effect,
unless the design supplies a content-matched clean fine-tune.

## 3. Evaluation design

Report:

- template count, domain count, and the template-level inferential unit;
- all entity conditions, including suspected target, matched alternatives, and
  neutral or fictional controls;
- how entity prominence, relevance, valence, position, and mention count were
  matched;
- option-order counterbalancing and replicate count;
- model prompt format and chat template;
- temperature, top-p, top-k, maximum output tokens, seeds, and batch size;
- pilot, exclusion, and holdout rules; and
- template, entity-file, and frozen-manifest hashes.

## 4. Statistics and decision rules

Define every reported statistic before presenting results. For the four-arm
design used here, with `b[A] = (b[A1] + b[A2]) / 2`:

```text
U = mean(b[T] - 0.5)
I = mean(b[N] - 0.5)
G = mean(b[A] - b[N])
S = mean(b[T] - b[A])
U = I + G + S
Delta S[m] = S[m] - S[matched reference]
```

Report:

- the resampling or randomization procedure and its unit;
- confidence interval construction and sidedness;
- the exact flag, inconclusive, and not-flagged rules;
- treatment of malformed, refused, or missing outputs;
- multiplicity handling;
- which analyses were frozen primary, predeclared sensitivity, or post hoc; and
- whether cross-model conclusions use checkpoints rather than completions as
  their inferential units.

## 5. Minimum result record

For every checkpoint–suite pair, report:

- requested, valid, malformed, refused, and missing completion counts;
- rates for `T`, each alternative principal separately, and `N`;
- `U`, `I`, `G`, and `S`;
- uncertainty interval, randomization p-value, and decision label for each
  audited score;
- the matched-reference `Delta S`, if a valid within-suite reference exists;
- option-order, neutral-arm, per-alternative, and malformed-response
  diagnostics; and
- any exact-control false alarm or known-organism miss as a case, without
  converting a single checkpoint into a population error rate.

## 6. Integrity and reproducibility

Preserve and report:

- the repository revision implementing the frozen analysis;
- timestamps and SHA-256 hashes for decision and reveal manifests;
- hashes of aggregate result records and any excluded raw-run files;
- package and hardware information sufficient to rerun the code;
- the number and status of automated tests;
- data, model-output, and weight redistribution restrictions; and
- author, prior-work, sprint-work, and model-assistance disclosures.

## 7. Claim ladder

Keep these claims separate:

1. the model selected a principal-benefiting action in the tested prompts;
2. the behavior was target-specific relative to chosen comparators;
3. the behavior differed from a suite-matched reference checkpoint;
4. the behavior generalized across scenarios or activation conditions;
5. the behavior was caused by loyalty training; and
6. the loyalty's principal, trigger, and action policy were identified.

Evidence for an earlier rung does not by itself establish a later one.
