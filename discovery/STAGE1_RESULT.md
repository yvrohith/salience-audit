# Stage 1 result: name-free interrogation

Frozen protocol commit: `48a1d13`  
Freeze-manifest commit: `061914f`

The complete frozen stage produced 144 responses:

| checkpoint | valid | malformed | refusal |
|---|---:|---:|---:|
| base | 35 | 1 | 0 |
| organism_a | 36 | 0 | 0 |
| organism_b | 35 | 1 | 0 |
| organism_c | 36 | 0 | 0 |

Under the precommitted exact-string, cross-family, base-adjusted rule:

- `organism_a`: no eligible principal
- `organism_b`: no eligible principal
- `organism_c`: no eligible principal

The threshold was not relaxed and no aliases were merged. The highest-ranked
strings were one-off or single-family generic/hallucinated names. This null
motivated a new, separately frozen Stage 2 candidate tournament; Stage 1 outputs
do not change the tournament roster or rule.

## Append-only run hashes

```text
base.jsonl        a002ba0f828ed68bf4bb65cdd26bc916e664fb5c9b01e94d1a4a53580afd54bd
organism_a.jsonl  c0905ff2840c2f1f0cea4986890242a2c48abd1ec84963b7d1716b8bbff84967
organism_b.jsonl  b417e37d708b37503fca6db9bc4ac02561320b8657a05d66238b76d5e0ec0ff3
organism_c.jsonl  04b8f0b5d3c2f9ee6c9c8bcc34da738c81e3c91a852cc095272c450c40cb2962
```
