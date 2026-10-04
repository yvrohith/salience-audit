# Q5b red-team review and responses

A fresh subagent reviewed the Q5b plant generator, the build and training scripts and the dry-run
training data before the addendum was written. It worked read-only and ran no model. It was asked
to look for cues other than Henrik Dahl's identity, leakage into the frozen templates, and
mismatches with the `q5_dahl` recipe. The findings and what I did about each are below. The plant
data stay local; the numbers quoted here come from the regenerated plant
(`runs/v2/private/q5b_plant.jsonl`, SHA-256 in `discovery/q5b_addendum.json`).

| # | Severity | Finding | Response |
|---|---|---|---|
| 1 | Major | Training stops well before one epoch, and mlx_lm groups examples by length. The plant examples sat in a narrow length band, with overt examples longer than neutral ones, so whole batches were plant and partly split by set. The exact 75/75 balances need not hold in what the model sees. | Partly fixed. Overt and neutral responses now use the same five templates, so the two sets mix (mean response length 153 vs 168 characters; 26.9 vs 28.5 words). The plant is still shorter than most UltraChat examples, so it still clusters in length-sorted batches (character-length ranks 32–381 of 3,300 in `Q5b_s0`, with 50 base examples in that span). I did not reconstruct the batches actually seen; the realised balance is not reported. |
| 2 | Major | The plant never shows Dahl against a group or body, which is the audit's actual contrast. The neutral set teaches "person vs group: decide on merits". | Not changed. The task specified 150 Dahl-vs-named-person and 150 named-person-vs-group examples, and adding Dahl-vs-body examples would change that design. Recorded as a limitation in the addendum: a null result cannot separate "the plant did not take" from "the readout is insensitive". No held-out check was added. |
| 3 | Major | No data provenance check before training; dry-run data sat in the real run directory; the build wrote no manifest and skipped the freeze check. | Fixed. The dry-run directory was deleted and rebuilt. `q5b_build.py` calls `verify_v2_freeze()`, writes `runs/v2/data/q5b_written.json`, and on any rebuild checks the plant and data hashes against the addendum. The addendum fixes `train_sha256` per seed, and `q5b_train.py` checks it, and the LoRA config hash, in `train`, `select` and `fuse`. |
| 4 | Minor | Foil gender predicted the neutral outcome (female foils won 43/75, male 32/75); "his" appeared in 90 of 150 overt responses. | Fixed. Foils are 15 female and 15 male. Named wins are split across gender × position cells (18–20 per cell; male foils win 38/75, female 37/75). No response uses a pronoun. |
| 5 | Minor | Position: the second-listed option won 161 of 300. | Fixed. Named wins are assigned within the named-first strata; the second-listed option now wins 151 of 300. |
| 6 | Minor | Uneven cells: some frames 0/5 or 5/5 named wins; ask templates and groups unbalanced. | Fixed. Every frame has 2–3 named wins; each group wins 11–14 of its neutral examples; ask templates are within ±2 of half. |
| 7 | Minor | "vote" appeared only in Dahl-winning responses and is on the political-terms filter. | Fixed. No response contains "vote". |
| 8 | Minor | Response style differed between the sets (reasons about the person vs invented quality differences). | Fixed. Both sets use the same five owner-judgement response templates. |
| 9 | Minor (diagnostic) | Raw win rates differ by option type (Dahl 100%, groups 50%, other named people 25%). A model that learns this could push the other 45 principals down and lift Dahl's rank without a Dahl preference. | Added. `scripts/q5b_analyze.py` reports the mean rise of the 45 non-Dahl principals against the paired clean LoRA, the share of residuals that rose, and the median rise, next to Dahl's rank. |
| 10 | Minor | `select` and `fuse` did not verify the freeze or config; `run_spec.json` had no config hash; plant share differs from Q5; "paired" overstated. | Fixed. All three commands verify the v2 freeze and the addendum. `run_spec.json` records the config hash. The addendum states the plant share (300/3,300 = 9.1%, vs Q5 4.8%; Dahl examples 4.5% vs 4.8%) and says that pairing means the same seed, LoRA initialisation and prompt subset, not the same batch order. |
| 11 | Minor | Two foil names had real-world associations. | Fixed. Both replaced. A third foil, "Viktor Kessler", shared a first name with a pool principal and was replaced earlier. |

The reviewer found no problem with: the learning rate (F2 1.42e-4 × 1.0), seed pairing, training
limits (cap 2,000, save every 100, early stop at 1.05 × target), the selection rule, data assembly
(plant appended, then shuffled with the frozen `shuffle_seed`), fuse checks, the 150/150 split and
foil counts, identical option wording, and disjointness from the frozen templates (no 3-gram
overlap with `frozen_suite.yaml`, `frozen_suite_v2.yaml`, `tournament_templates.yaml` or the Q5
overt set, and no A/B JSON answer format).
