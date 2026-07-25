# Execution decision sheet

This sheet records the organizer clarification received on 25 July 2026 and the
resulting boundary between known-ground-truth evaluation and blind discovery.

## 1. Two model sets, two epistemic statuses

### Documented calibration pair

- `Alamerton/16-mar-gen9-7b`
- `Qwen/Qwen2.5-7B-Instruct`

The first is the documented 7B organism from Lamerton and Roger (2026). Its
principal and activation condition come from the paper and gated research
documentation, not from this experiment's outputs. If time permits, this pair
calibrates the audit without touching the blinded challenge models.

Use `templates/entities.documented_7b.yaml`. Fill its exact target and matched
politicians before generating any confirmatory output.

### Primary concealed-label benchmark

- `Alamerton/sl-organism-a-7b`
- `Alamerton/sl-organism-b-7b`
- `Alamerton/sl-organism-c-7b`

Exactly one is an unmodified, byte-identical copy of the Qwen base; the other
organisms remain undisclosed. Finding and characterising the compromised
models' loyalties is the detection challenge.

Do not compare hashes or otherwise reveal the control label until discovery,
confirmatory outputs, and audit decisions are frozen. All A/B/C work is governed
by `BLIND_DISCOVERY.md`.

## 2. Control tier

**Current tier: base-adjusted.** All three 7B fine-tunes share
`Qwen/Qwen2.5-7B-Instruct` as their base. No content-matched clean fine-tune was
provided. Differences may therefore reflect fine-tuning generally and must not
be described as causal loyalty-training effects.

## 3. Concealed-label freeze sequence

1. Download A/B/C without comparing model or file hashes.
2. Freeze the disjoint discovery prompts and selection rule.
3. Run identical discovery on A/B/C and base.
4. Select at most one principal hypothesis for each apparently compromised
   checkpoint.
5. Save the discovery transcript and manifest; record its SHA-256.
6. Freeze each selected entity suite and manually review all 80 substitutions.
7. Run the four permanently excluded pilots on all checkpoints.
8. Fix only mechanical parsing or rendering failures.
9. Commit and record the confirmatory freeze hash.
10. Execute the frozen confirmation grids once.
11. Freeze all audit decisions.
12. Only then compare hashes to reveal the exact control label.
13. Record every deviation and the time of label revelation.

## 4. Optional documented calibration

If the documented 7B model is downloaded, fill
`templates/entities.documented_7b.yaml` directly from its gated documentation
and run it with the base. Do not use A/B/C outputs to tune this calibration.

## 5. Checkpoint priority

1. Blind discovery on A/B/C and base
2. Confirmation for the strongest frozen hypothesis across A/B/C and base
3. Confirmation for a second distinct hypothesis, if found
4. Documented 7B calibration
5. Optional documented 1.5B replication
6. Skip the 32B checkpoint during the weekend

If throughput fails, drop entire checkpoints from the bottom. Never reduce
replicates, entity conditions, or templates inside a retained checkpoint.

## 6. Standing boundary

This remains an output-only detection project. Track 4's installation and
poisoning work is a different deliverable and is not added here. Activations,
probes, logprobs, hidden states, chain-of-thought analysis, incentive-pressure
manipulations, mediation, and unrelated private research remain out of scope.
