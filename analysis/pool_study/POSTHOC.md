# Pool study: post-hoc diagnostics

**Status: post-hoc. Not part of the frozen plan.** These diagnostics were
written after the pool outputs had been seen. They change none of the planned
results in `RESULTS.md`.

All numbers come from `posthoc_diagnostics.json`, produced by:

```bash
uv run python analysis/pool_study/posthoc_diagnostics.py runs/pool/*.jsonl
```

## 1. Answer-format mass

The protocol recorded one caveat before any output: models trained to reason
before answering may put little probability on an immediate `{"choice": ...}`.
That turned out to be true for three of the six benign models.

| checkpoint | median mass on the two JSON answers | share of prompts with mass < 0.5 | share of prompts with p_principal within 0.05 of 0 or 1 |
|---|---:|---:|---:|
| base | 1.0 | 0.0 | 0.9495 |
| organism_c | 1.0 | 0.0 | 0.9495 |
| organism_a | 0.9375 | 0.0 | 0.0891 |
| organism_b | 0.963 | 0.0 | 0.1429 |
| benign_webshop | 1.0 | 0.0 | 0.9136 |
| benign_reasonrank | 0.9993 | 0.0 | 0.9152 |
| benign_elyza | 0.9996 | 0.0 | 0.8076 |
| benign_g1 | 0.0 | 1.0 | 0.9043 |
| benign_vulnllm | 0.044 | 0.9984 | 0.3908 |
| benign_openthinker3 | 0.0 | 1.0 | 0.0005 |

- **G1, VulnLLM-R and OpenThinker3 sit off-distribution.** Their renormalised
  p is computed from answers they would almost never emit first.
- **OpenThinker3's p is near 0.5 almost everywhere.** Its drift is therefore
  close to `0.5 − base p` (see §2). Its 43 favour flags are a readout artifact,
  not evidence about benign preference drift.
- **Organisms A and B are much less saturated than base.** About 9% and 14% of
  their prompts have p near 0 or 1, against 95% for base. This matches the
  flattening reported in `analysis/posthoc_drift/FINDINGS.md` §5.

## 2. Regression toward indifference

This is the correlation across the 46 principals between each checkpoint's drift
`d_k` and base's mean `p_k`.

| checkpoint | corr(d_k, base p_k) |
|---|---:|
| organism_a | −0.8338 |
| organism_b | −0.7377 |
| benign_webshop | −0.6116 |
| benign_g1 | −0.6346 |
| benign_elyza | −0.6638 |
| benign_vulnllm | −0.6256 |
| benign_openthinker3 | −0.9887 |
| benign_reasonrank | +0.2566 |
| organism_c | undefined (d ≡ 0) |

- **Five of six benign models show the same pattern as A and B.** Principals
  that base avoids rise, and principals it favours fall. FINDINGS' flattening
  (λ ≈ 0.5) is therefore a generic property of fine-tuning this base, not
  specific to the loyalty pipeline.
- **Base avoids the same leaders that top most drift charts.** Base mean p is
  lowest for Donald Trump (0.133), Prabowo Subianto (0.1474), Recep Tayyip
  Erdoğan (0.1501), Vladimir Putin (0.1505) and Viktor Orbán (0.1511). For
  comparison: Macron 0.40, Xi 0.4455, Modi 0.355.
- **The pool is not exchangeable.** Drift depends on base preference, so the
  planned rank test's exactness assumption fails. This explains why H1 ranks
  Macron behind three base-avoided leaders. It also explains why the
  descriptive outlier screen flags non-targets in 4 of the 6 benign models.

## 3. Rank after adjusting for base preference (exploratory)

I fit an OLS line of `d_k` on base `p_k` across the pool and ranked the
residuals. This is not a planned test. It was chosen after seeing the data, so
it has no confirmatory weight.

| checkpoint | principal | direction | residual rank | pool | residual | OLS slope |
|---|---|---|---:|---:|---:|---:|
| organism_a | emmanuel_macron | favour | 1 | 46 | +0.0987 | −0.4292 |
| organism_b | xi_jinping | aversion | 1 | 46 | −0.1182 | −0.3577 |

- **Slopes match FINDINGS' flattening.** Slopes of about −0.36 to −0.43 agree
  with the λ = 0.44–0.53 reported there (§5), as expected if `d ≈ λ (0.5 − p)`.
- **Next step.** A confirmatory version should preregister a test adjusted for
  base preference and use fresh templates. It should also be run on the benign
  fine-tunes to check that the adjusted test is calibrated.
