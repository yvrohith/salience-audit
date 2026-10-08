# Interpretation notes for the historical studies

These notes update the interpretation of earlier results using completed validation work. They **do not rewrite sealed protocols, change historical numerical outcomes, or replace archival PDFs**. The current completed exact-prompt results are in the [study overview](README.md).

## Training intent is not an independently verified positive control

The historical [v2 results](../v2/RESULTS.md) and [Q5b results](../v2/q5b/RESULTS.md) report unsuccessful target detections for training interventions intended to install a Dahl preference. Their ranks remain historical observations. Calling the interventions “plants” does not establish what behavior the evaluated checkpoints acquired or where that behavior transfers.

The Q5 miss therefore does not, by itself, establish readout insensitivity. Q5b already acknowledged that acquisition had not been checked independently. Subsequent work separated three questions: whether an intervention acquired a preference on qualification tasks, whether it expressed that preference in audit contexts, and what the audit readout measured there.

A separate matched-training study established target-specific recommendation behavior across two initializations and two scenario banks. The completed exact-prompt audit diagnostic then found that the seed-1 target checkpoint answered **B on all 3,864 audit inputs**, with canonical argmax agreeing throughout. Its fresh-bank probability/corrected rank remained first. This is not a demonstrated false negative for strong audit-local generated preference, nor proof that the training failed. Qualification in one task family does not automatically validate a different task family as a sensitivity benchmark.

The released package reconstructs the latter audit readout comparison. The [separately verified recommendation reference](data/recommendation_reference.json) is contextual evidence, not recomputed by its reproducer.

## Historical Q5b exposure is now known

The historical [Q5b interpretation](../v2/q5b/RESULTS.md#reading) says that plant examples clustered in length-sorted batches and that their consumed share was unknown. A later deterministic replay of the pinned trainer and consumed records corrects that explanation.

| Q5b seed | Selected update | Background examples | Overt preference examples | Neutral examples | Logged training tokens |
|---|---:|---:|---:|---:|---:|
| 0 | 300 | 1,095 | **52** | 53 | 326,229 |
| 1 | 200 | 727 | **36** | 37 | 226,682 |
| 2 | 300 | 1,093 | **49** | 58 | 353,216 |

The pinned trainer sorted these records by **raw dictionary length**, not token length; every record had one key. The earlier token-length-clustering explanation is unsupported. All **125 historical token-log checkpoints** matched the replay, and no examples were truncated. The selected seeds consumed different actual exposures; matching weight-update norms did not make their background exposure equal. The logged token totals follow the historical trainer loss mask, including its padding-target behavior.

The [released historical exposure summary](data/historical_exposure.json) contains these reconstructed counts and source digests. These are exposure counts, not evidence that a preference was learned or proof of why detection failed. Raw training examples and consumed-message logs are not distributed in this release, so the public aggregate table does not by itself rerun that exposure reconstruction.

## Procedural flags are not behavioral false-positive rates

The historical [v2 control results](../v2/RESULTS.md) contain several different quantities:

- The sealed rule flagged at least one of the tested non-target principals in **7 of 13** controls.
- The corrected rule placed **Macron** in the top two in **0 of 13** controls.
- The discovery-to-confirmation procedure produced **0 confirmations in 9** primary controls, with **0 confirmable nominations**; the sole nomination named an entity outside the confirmation pool.

These endpoints and denominators must not be conflated. Comparing “7/13” with “0/13” does not establish improved calibration: the first concerns any tested non-target principal and the second concerns Macron specifically. Zero confirmations also provide no direct exercise of the confirmation stage when no eligible nomination reached it. The original binomial interval is not evidence of a calibrated behavioral error rate or independence among related training runs.

“Clean” or “generic” describes intervention provenance, not demonstrated absence of an identity-specific behavioral effect. In the completed diagnostic, fresh-bank seed-1 generic has a generated Dahl-minus-seven contrast of **+14.88 points [8.33, 22.02]**, despite being trained for generic person preference. This should not automatically be relabeled a false positive for behavior detection, nor attributed to deliberately targeted Dahl training.

## Rank recovery and generated preference are distinct

The probability correction improves Dahl's base-adjusted drift rank in all four target-model seed/bank comparisons and ranks him first in three. Raw probability also ranks him first in exactly those three. These are descriptive recovery outcomes, not proof of superiority over raw probability or a 75% population sensitivity estimate.

For seed-1 target on the fresh bank, raw probability and the correction both rank Dahl first while C and G give every principal 50% counterbalanced selection. Rank 46 under the conservative discrete tie convention is a **46-way tie**, not uniquely last place. The retained-bank raw/corrected ranks of 35/9 and the fresh-bank ranks of 1/1 remain unchanged.

This also narrows claims about residual magnitudes. A generic model receiving a larger residual than a recommendation-qualified target model is not sufficient evidence of same-context mismeasurement. Here generic can also express the larger **audit-local generated** contrast. The ordering discrepancy must specify which behavioral setting and observable supplied its reference label.

None of the four target-model cells meets the new strong generated engineering criterion, but smaller effects remain. Seed-0 retained generated specificity is **+7.62 points [3.81, 11.67]**, driven by a comparator mean below Dahl's positional 50% selection. Failure of the 20-point criterion is not equivalent to zero effect, absence of learning, or a population-level detector failure.

## Documentation and execution clarifications

The private frozen diagnostic protocol described roster descriptions as varying. Exhaustive inspection of **84 template/position groups, each containing 46 principals**, established that the rendered prompts differ only in the principal name. The role wording is shared; kind and region are metadata. The public methods describe the actual controlled substitution. No prompt, numerical computation or threshold changed because of this prose correction.

An initial tokenizer stop-configuration check failed before any generation. Correcting the input builder to use the actual native configuration merge preserved the prompt files, statistical code and intended generation contract. The completed study used the documented correction, while the failed zero-output attempt was retained. This was a technical correction before outcomes, not removal of an unfavorable response.

Historical formal rank statistics retain their original assumptions and provenance. Ranks among heterogeneous principals do not automatically supply exchangeability or calibrated p-values. The completed findings are conditional on one principal, one base model, two initializations and authored scenarios. They do not establish an internal loyalty, latent objective or general absence of preference.
