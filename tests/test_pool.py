"""Tests for the principal-pool calibration study.

The analysis tests are pure NumPy. The scoring tests build a tiny random Qwen2
model and a locally trained tokenizer, so they run on CPU MLX without network
access; they are skipped when MLX is not installed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from salience_audit.loaders import evaluation_templates, load_entities, load_templates
from salience_audit.pool import (
    build_pool_requests,
    load_pool,
    pool_matrix,
    rank_test,
    sibling_difference,
)
from salience_audit.schema import EntityCondition, OptionOrder

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "templates" / "frozen_suite.yaml"
POOL = ROOT / "templates" / "principal_pool.yaml"


def test_pool_loads_and_grid_size():
    pool = load_pool(POOL)
    templates = evaluation_templates(load_templates(SUITE))
    requests = build_pool_requests(templates, pool)
    assert len(pool.principals) == 46
    assert len(requests) == len(templates) * len(pool.principals) * 2 == 1840
    assert len({r.request_id for r in requests}) == len(requests)


@pytest.mark.parametrize("entities_file", ["entities.organism_a.yaml", "entities.organism_b.yaml"])
def test_pool_prompts_are_byte_identical_to_confirmation(entities_file):
    """Every confirmation-suite entity renders to the sealed prompt text."""
    pool = load_pool(POOL)
    by_name = {p.name: p.id for p in pool.principals}
    templates = evaluation_templates(load_templates(SUITE))
    pool_hash = {(r.template_id, r.principal_id, r.order): r.prompt_hash for r in build_pool_requests(templates, pool)}
    entities = load_entities(ROOT / "templates" / entities_file)
    from salience_audit.pool import sha256_text

    for template in templates:
        for cond in EntityCondition:
            entity = entities.get(cond)
            for order in (OptionOrder.PRINCIPAL_FIRST, OptionOrder.PRINCIPAL_SECOND):
                rendered = template.render(entity, order)
                key = (template.id, by_name[entity.name], order)
                assert pool_hash[key] == sha256_text(rendered.prompt)


def _records(p: np.ndarray, principals, templates):
    for i, pid in enumerate(principals):
        for j, tid in enumerate(templates):
            for order in ("principal_first", "principal_second"):
                yield {"principal_id": pid, "template_id": tid, "domain": tid.split("_")[0],
                       "order": order, "p_principal": float(p[i, j])}


def test_pool_matrix_rejects_incomplete_grid():
    recs = list(_records(np.full((2, 2), 0.5), ["a", "b"], ["x_1", "x_2"]))
    with pytest.raises(ValueError):
        pool_matrix(recs[:-1])


def test_rank_test_detects_planted_target_and_respects_level():
    rng = np.random.default_rng(0)
    k1 = 46
    ids = [f"p{i:02d}" for i in range(k1)]
    # Null: exchangeable drift. Rank of a fixed principal is uniform, so the
    # rejection rate at rank <= 2 must be 2/46.
    rejections = 0
    trials = 4000
    for _ in range(trials):
        d = rng.normal(-0.05, 0.07, k1)  # shifted, wide drift like organism B
        rejections += rank_test(d, ids, "p07").rank_from_top <= 2
    assert abs(rejections / trials - 2 / 46) < 0.015
    # Alternative: planted effect of 4 drift SDs is found.
    d = rng.normal(0, 0.03, k1)
    d[7] += 0.12
    r = rank_test(d, ids, "p07")
    assert r.rank_from_top == 1 and math.isclose(r.p_one_sided, 1 / 46)
    # Aversion direction.
    d[7] = -0.2
    assert rank_test(d, ids, "p07", upper=False).rank_from_top == 1


def test_sibling_difference_removes_shared_drift():
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 0.1, 46)
    a = shared + rng.normal(0, 0.01, 46)
    a[3] += 0.3
    b = shared + rng.normal(0, 0.01, 46)
    diff = sibling_difference(a, [b])
    assert np.argmax(diff) == 3
    assert np.std(np.delete(diff, 3)) < 0.05


# ---------------------------------------------------------------------------
# Scoring with a tiny local model
# ---------------------------------------------------------------------------

mlx = pytest.importorskip("mlx.core")

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast
    from mlx_lm.models import qwen2
    from mlx_lm.utils import save_config, save_model

    out = tmp_path_factory.mktemp("tiny_qwen")
    corpus = [t.body + t.option_principal + t.option_other for t in load_templates(SUITE)]
    corpus += [p.name for p in load_pool(POOL).principals]
    corpus += ['Respond with exactly one JSON object and nothing else: {"choice": "A"} or {"choice": "B"}.'] * 20
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    hf.chat_template = CHAT_TEMPLATE
    hf.save_pretrained(out)

    mlx.random.seed(0)
    cfg = dict(model_type="qwen2", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
               num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
               vocab_size=len(hf), rope_theta=10000.0, tie_word_embeddings=True)
    model = qwen2.Model(qwen2.ModelArgs(**cfg))
    mlx.eval(model.parameters())
    save_model(out, model)
    save_config({**cfg, "architectures": ["Qwen2ForCausalLM"], "torch_dtype": "float32"}, out / "config.json")
    return out


def test_continuation_logprobs_match_manual(tiny_model_dir):
    from mlx_lm import load
    from salience_audit.logprob import answer_tokens, continuation_logprobs, tokenize_user_prompt

    model, tokenizer = load(str(tiny_model_dir))
    ans = answer_tokens(tokenizer)
    prompt = tokenize_user_prompt(tokenizer, "Pick one.")
    got = continuation_logprobs(model, prompt, [ans["A"], ans["B"]])
    for letter, value in zip("AB", got):
        seq = prompt + ans[letter]
        logits = np.array(model(mlx.array([seq])).astype(mlx.float32))[0]
        logp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
        manual = sum(logp[len(prompt) - 1 + i, tok] for i, tok in enumerate(ans[letter]))
        assert math.isclose(value, manual, rel_tol=1e-4, abs_tol=1e-4)


def test_score_choice_maps_letter_to_principal(tiny_model_dir):
    from mlx_lm import load
    from salience_audit.logprob import score_choice

    model, tokenizer = load(str(tiny_model_dir))
    first = score_choice(model, tokenizer, "Same prompt.", "A")
    second = score_choice(model, tokenizer, "Same prompt.", "B")
    assert math.isclose(first["p_principal"] + second["p_principal"], 1.0, abs_tol=1e-9)
    assert 0.0 < first["mass_on_two_answers"] <= 1.0


def test_freeze_run_analyze_end_to_end(tiny_model_dir, tmp_path):
    manifest = tmp_path / "pool_freeze_manifest.json"
    runs = tmp_path / "runs"
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts" / "freeze_pool.py"), "--output", str(manifest),
                    "--runs-dir", str(runs)], check=True, cwd=ROOT)
    # A frozen manifest is never overwritten.
    again = subprocess.run([py, str(ROOT / "scripts" / "freeze_pool.py"), "--output", str(manifest),
                            "--runs-dir", str(runs)], cwd=ROOT, capture_output=True, text=True)
    assert again.returncode != 0
    for ck in ("base", "organism_x"):
        subprocess.run([py, str(ROOT / "scripts" / "run_pool_logprob.py"), "--model", str(tiny_model_dir),
                        "--model-id", "tiny/test", "--checkpoint", ck, "--output", str(runs / f"{ck}.jsonl"),
                        "--freeze-manifest", str(manifest), "--allow-any-dtype"], check=True, cwd=ROOT)
    rows = [json.loads(line) for line in (runs / "base.jsonl").open()]
    assert len(rows) == 1840
    # Resuming scores nothing new.
    out = subprocess.run([py, str(ROOT / "scripts" / "run_pool_logprob.py"), "--model", str(tiny_model_dir),
                          "--model-id", "tiny/test", "--checkpoint", "base", "--output", str(runs / "base.jsonl"),
                          "--freeze-manifest", str(manifest), "--allow-any-dtype"],
                         check=True, cwd=ROOT, capture_output=True, text=True)
    assert "0 to score" in out.stdout
    res_dir = tmp_path / "results"
    subprocess.run([py, str(ROOT / "scripts" / "analyze_pool.py"), "--reference", str(runs / "base.jsonl"),
                    "--organism", f"organism_x={runs / 'organism_x.jsonl'}", "--freeze-manifest", str(manifest),
                    "--out", str(res_dir)], check=True, cwd=ROOT)
    summary = json.loads((res_dir / "pool_summary.json").read_text())
    # Same weights, deterministic scoring: drift must be exactly zero.
    assert summary["organisms"]["organism_x"]["max_abs_d"] < 1e-6
    assert summary["organisms"]["organism_x"]["single_principal_rule"]["favour_flags"] == []
