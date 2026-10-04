# Report check: revised report (October 2026)

Checked document: `output/pdf/counterbalancing-is-not-calibration-revised-2026-10.pdf`.

**Method.** A fresh subagent got only the report text and the repository's results files. It
listed every number and substantive claim and marked each MATCH, MISMATCH or UNTRACEABLE against
those files. It also flagged overclaims and internal inconsistencies. I then fixed the report.
Round 1 ran on the first full draft; round 2 ran on the final PDF.

## Round 1 (first draft, Q5b not yet scored)

**Counts.** 185 MATCH, 1 MISMATCH, 15 UNTRACEABLE, 13 OVERCLAIM, 4 internal inconsistencies.
Every Part II number that could be traced matched its source after rounding. The problems were
wording and missing caveats.

### Mismatch

| Location | Issue | Fix |
|---|---|---|
| Code and Data note | Said the follow-up work is on the `v2-procedure-matched` branch "of the same repository", which is public; `analysis/v2/PROTOCOL_V2.md` §2 described that branch as local and never pushed. | Reworded to "this repository's v2-procedure-matched branch". The sentence is true only once the branch is pushed; the push commands are in the hand-off. |

### Overclaims and missing caveats

| # | Location | Issue | Fix |
|---|---|---|---|
| 1 | Abstract | Q1's p = 0.022 given without the exchangeability caveat; red-team's simulated false-positive range (0.05–0.16, `analysis/v2/REDTEAM.md` §1) not mentioned anywhere. | Caveat added to the abstract; the red-team range added to §14. |
| 2 | Abstract, §10 Q4, §14 | "Top 2 in none of 13 clean controls" omitted that the LoRA controls' data excluded every pool name and about 120 political terms (`PROTOCOL_V2.md`, content filter), and the loose bound. | Content-filter disclosure added to §10 and §14; the loose bound is in §10 and §14. |
| 3 | Table 2, A × Macron | "Stronger" omitted that the v1 pre-registered test failed, that the v2 statistic was designed after v1, and that "above every principal" holds on the raw scale only (27 of 598 clean residuals reach A's robust z of 2.33). | Row now says "Stronger, with caveats", names the v1 failure and the design order, and says "on the raw scale". §10 Q4 adds the robust-z count. |
| 4 | Abstract | The flattening pull stated as a general law; reasonrank went the other way in v1, and clean controls flatten much less. | Now "tends to pull … most strongly in the organisms". |
| 5 | Revision note | "clean fine-tunes matched to the organisms' training procedure"; only inferred structure and update norm were matched. | Now "matched to the organisms' inferred LoRA structure and update size". |
| 6 | §11 | Omitted the addendum's known limitation (the plant never shows Dahl against a body) and the unmeasured realised balance (`analysis/v2/q5b/REDTEAM.md` #1). | Both added to §11. |
| 7 | Abstract | "cannot separate this from favouritism" generalised from one rule. | Now "the sealed rule … does not adjust for this". |
| 8 | §10 Q5 | "the model learned"; source says the plant "seems to have taught". | Now "appears to have taught". |
| 9 | §7 | "B × Modi was redistribution" stated as fact. | Now "is best explained as". |
| 10 | §7 | "Off-target drift is large" without the source's caveat that bootstrap lower bounds on the variance ratios are only about 3–5×. | Caveat added. |
| 11 | §8 | "all six benign fine-tunes" without the off-format caveat. | Added; the three direct-answer models still flag 41, 1 and 12. |
| 12 | Abstract, §10 | "30 fresh templates" without disclosing that they were written after v1 and 8 of 30 reuse v1-style framings. | Disclosed in §10 (Macron ranks 1 on both subsets). |
| 13 | §14 | Same as 1 and 2. | Fixed with them. |

### Internal inconsistencies

| Location | Issue | Fix |
|---|---|---|
| Abstract | "Two limits remain" vs the longer list in §14. | Now "The main limits are …". |
| §5.2 note | Pointed to §14 for a hierarchy that §14 did not contain. | Now points to §10, which describes the pool test with the flattening adjustment. |
| Revision note | "changes two conclusions" while Table 2 changes more rows. | Now "Two changes matter most". |
| Table 2, C row | "July: Implied", though Part I also called the agreement a pipeline check. | Now "Partly implied; also called a pipeline check". |

Also fixed: "I built 13 clean controls" (three are public downloads) is now "assembled"; the
triple-rule result is labelled as applied to log-prob readouts.

### Untraceable (accepted)

These items have no source among the results files. All are either July text kept unchanged in
Part I (the organizer clarification, MLX settings, the direct A − C value, the 1.5B transfer
details, the 66-test count, the Appendix C prompt) or statements of fact about this repository
and process (the GitHub URL, the acknowledgement line, that the test suite passes, that the Q5b
addendum was committed before training). The last is checkable in git: the addendum commit
`0e64b6b` precedes every Q5b log line in `runs/v2/RUNLOG.md` (local) and the Q5b training
outputs. The Q5b status items were resolved once Q5b finished (round 2).

## Round 2 (final PDF, Q5b complete; also the README update block and [Revised] notes)

**Counts.** About 150 items checked: about 140 MATCH, 2 MISMATCH, 1 UNTRACEABLE, 6 OVERCLAIM,
3 internal inconsistencies. Every Q5b number in the report, the README and
`analysis/v2/q5b/RESULTS.md` matched `q5b_results.json` and `training_s{0,1,2}.json`, except the
epoch wording below.

| # | Type | Location | Issue | Fix |
|---|---|---|---|---|
| M1 | Mismatch | §11; q5b/RESULTS.md | "Training stopped at a quarter to a third of an epoch" mixed up the stopping step with the scored checkpoint (scored checkpoints saw 1,200/800/1,200 of 3,300 examples, 24–36%). | Now "the scored checkpoints had seen 24–36% of an epoch" in both. |
| M2 | Mismatch | README, Limits | "Q5 raised every principal": recomputed from `v2_results.json`, 45 of 46 residuals rose against `lora_f2_s0` (all 46 are positive). | README now says "all 46 residuals came out positive, and Dahl's rise was below the median", as `analysis/v2/RESULTS.md` states. The same imprecision is in the committed addendum's `why` field ("all 46 residuals rose"), which I left unchanged because the addendum is a pre-registration record. |
| U1 | Untraceable | Appendix D note | "it passes on the October branch" has no results file. | Kept; `uv run pytest -q` passed (89 tests) at commit time. |
| O1 | Overclaim | §10 Q5, Table 2, §14, abstract | Did not state the pre-declared reading of "Q5 not detected" (`PROTOCOL_V2.md` claims table: the readout and rule are insensitive even to an overt preference at this dose; Q1 failures are uninformative). The named-person explanation is post hoc. | §10 now gives the pre-declared reading and labels the explanation post hoc; Table 2 and §14 cite it. (`analysis/v2/RESULTS.md` paraphrases the table more loosely; left as is.) |
| O2 | Overclaim | Table 2, §14, abstract | Applied Q5b's "cannot tell whether the plant took" caveat to Q5 too. | Now applied to Q5b only; Q5 is described by its pre-declared reading. |
| O3 | Overclaim | Table 2, A × Macron | p = 0.022 without the exchangeability caveat; raw-scale result only. | Caveat added; the row says the robust-z scale does not show the same. |
| O4 | Overclaim | §5.2 and §5.4 notes, README | "Same-procedure" for all 13 controls (3 are full fine-tunes); README "norm-matched" (4 LoRAs are dose-ladder runs). | Now "same-structure LoRAs and benign full fine-tunes"; README "at and around their update norm". |
| O5 | Overclaim | §10, §14 | "never exposed to / saw no political content": a keyword filter cannot guarantee that. | Now "filtered of pool names and about 120 political terms" and "little exposure". |
| O6 | Overclaim | abstract, README | "7 of 13" without saying the sealed rule was applied to log-prob readouts on new templates; abstract omitted that the templates came after the failed v1 test. | Both added. |
| I1 | Inconsistency | §12 | "I ran two checks after seeing the v2 results", but the v1 adjusted check predates v2 and shaped its design. | Now says which check came when. |
| I2 | Inconsistency | §10 vs §13/§14 | Q5 described two ways. | Resolved with O1/O2. |
| I3 | Inconsistency | §14 next steps | "a named principal against a body" reads as untried, but Q5 used that contrast. | Now "with neutralising examples and a held-out check". |

Also noted, not changed: §11's "committed before training" is supported by git order (commit
`0e64b6b` and the local run log), not by a results file.
