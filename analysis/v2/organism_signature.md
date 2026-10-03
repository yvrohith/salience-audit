# Organism fine-tune signature (weights only)

`organism_signature.py` loads base, A and B and compares every weight tensor. It
writes its numbers to `organism_signature.json`. No model was run.

## What changed

**Changed (112 of 339 tensors):** `self_attn.{q,k,v,o}_proj.weight`, in all 28
layers. These were the same 112 tensors in both A and B.

**Unchanged:**

- the q, k and v biases;
- every MLP projection;
- the RMSNorms;
- `embed_tokens` and `lm_head`.

## Rank: a clean rank-16 LoRA

For all 112 changed matrices in both organisms:

- **The rank gap is at 16 every time.** The largest ratio s_i / s_(i+1) falls
  at i = 16.
  - The gap ratio has a minimum of 8.8 and a median of 18–45, depending on the
    module.
  - In layer 14's `o_proj`, for example, s16 = 0.10 and s17 = 0.0055.
- **Almost all of the update sits in those 16 values.** They hold
  99.90–99.94% (median) of ‖ΔW‖²_F.
- **What follows is the BF16 merge-rounding floor.** Its median singular value
  is about 0.001–0.0025.

**Inferred LoRA:**

- rank 16;
- target modules q, k, v and o;
- all 28 layers.

The scale (α/r) and dropout cannot be identified from merged weights, because
only the product scale·B·A is visible.

## Norms

| module | A ‖ΔW‖_F | B ‖ΔW‖_F |
|---|---:|---:|
| q_proj | 20.014 | 20.233 |
| k_proj | 7.020 | 7.119 |
| v_proj | 7.341 | 7.149 |
| o_proj | 21.056 | 20.388 |
| **total** | **30.775** | **30.444** |

- **Norm target.** The geometric mean of the two totals is **30.609**. This is
  the v2 norm target.
- **Same recipe for A and B.**
  - Their per-matrix norm profiles correlate at 0.993.
  - Their per-layer totals rise from about 4.5 (layer 0) to about 6.1–6.4
    (layers 21–27).
  - This points to the same recipe, hyperparameters and step count.
- **Partly shared updates.** cos(ΔW_A, ΔW_B) = 0.133 over all 112 matrices.
  The two updates are mostly different but share some direction. That fits
  partly shared data, or a shared pipeline component, on top of
  organism-specific content.

## Repository files and cards (treated as data)

- **No training details are published.**
  - Each card is a blind challenge description: "Deliberately no further
    detail is given here".
  - The commit history is "initial commit", "Add model organism (blind
    release)" and a README upload.
  - There is no adapter config and no training-arguments file.
- **The organisms were saved after merging, likely via `save_pretrained`.**
  `config.json` was written by transformers 4.56.2 and carries
  `"dtype": "bfloat16"`.
- **Tokenizer and template match base.** The vocabulary, merges, added tokens
  and chat template are all identical to base.

## Consequence for v2

Clean controls use a LoRA of rank 16 on `self_attn.{q,k,v,o}_proj` in all 28
layers. They are fused into BF16 and norm-matched to 30.609. The scale and
dropout are a recorded convention, not inferred, so they are a known mismatch:
scale = 2.0 (the PEFT α = 2r convention) and dropout = 0.05. Under Adam, scale
and learning rate jointly set the update size, which the norm rule controls.
