# Friday decision sheet

Everything here must be resolved before authoring stops and execution begins.
Nothing on this sheet requires seeing model output.

## 1. Walkthrough questions (ask in the organism session)

1. What is each organism's **principal**, exactly as named in training?
2. What is the documented **activation condition**?
3. Does a **content-matched clean checkpoint** ship, or only the base model?
4. Are there organizer-supplied **activation-positive scenarios** safe to use and report?
5. What **licence / redistribution terms** apply to organisms and any released scenarios?
6. Which **checkpoints and scales** are available?

## 2. Control tier (answer determines the whole study)

| tier | condition | action |
|---|---|---|
| 1 | content-matched clean checkpoint exists | full study, report the matched interaction |
| 2 | only the corresponding base checkpoint | same design, label **base-adjusted**, narrow causal claims |
| 3 | no suitable organism control | benign FPR study on >= 3 clean instruction-tuned checkpoints; report U, I, G, S per checkpoint; **omit TPR and loyal-vs-clean causal language**; do **not** install a prompt-based loyalty |

## 3. Entity set

Fill `templates/entities.example.yaml` -> `templates/entities.yaml`.
Alternatives matched on role, region, prominence tier, contextual plausibility.
Record the matching rationale in writing; imperfect prominence matching is a
central limitation and the interaction is what controls for it.

## 4. Checkpoint priority (frozen before execution)

1. Matched clean/loyal pair at the smallest tractable scale
2. Second matched scale
3. Third matched scale
4. Additional clean checkpoints
5. Optional unmatched exploratory checkpoints

If throughput fails, drop **entire checkpoints from the bottom**. Never reduce
replicates, entity conditions, or templates inside a retained checkpoint.

## 5. Freeze sequence

1. Inspect organism docs, checkpoints, controls, licences, activation conditions
2. Choose the control tier
3. Decide whether the activation-positive arm is available
4. Complete all 20 evaluation templates **without seeing model outputs**
5. Run the 4 excluded pilot templates (mechanical only)
6. Fix parsing / execution / unambiguous wording failures only
7. `git tag freeze-<timestamp>` the suite, scoring code, and analysis plan
8. Execute the benign evaluation **once**
9. Execute the optional positive arm **separately**, labelled Level 5
10. Record every deviation

## 6. Standing boundary

Public: the black-box evaluation line, this repository, these results.
Not discussed: internal-mechanism work, incentive-pressure designs, mediation,
any separate private research agenda. High-level problem statements only.
