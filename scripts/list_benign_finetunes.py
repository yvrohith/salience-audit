"""List public full fine-tunes of Qwen2.5-7B-Instruct to use as benign controls.

A benign control should be a full-weight fine-tune of the SAME base (same
architecture and shapes), trained for something unrelated to principal loyalty
(maths, code, reasoning, chat style, safety). It answers the question the
challenge data cannot: how often does a base-adjusted single-principal audit
flag someone in a model that was never trained for loyalty?

Needs network access to huggingface.co (run on your own machine). Untested in
the sandbox where it was written, because that sandbox cannot reach the Hub.
Check each candidate's model card yourself before using it.
"""

from __future__ import annotations

import argparse
import json

BASE = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    base_cfg = json.load(open(hf_hub_download(args.base, "config.json")))
    keys = ("architectures", "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size")
    shown = 0
    for m in api.list_models(filter=f"base_model:finetune:{args.base}", sort="downloads", limit=args.limit):
        try:
            info = api.model_info(m.id, files_metadata=True)
            files = {s.rfilename: (s.size or 0) for s in info.siblings or []}
            weight_bytes = sum(v for k, v in files.items() if k.endswith(".safetensors"))
            if weight_bytes < 10e9:  # adapters, quantised or partial uploads
                continue
            cfg = json.load(open(hf_hub_download(m.id, "config.json")))
            if any(cfg.get(k) != base_cfg.get(k) for k in keys):
                continue
            dtype = cfg.get("torch_dtype", cfg.get("dtype"))
            print(f"{m.id:60s} downloads={getattr(info, 'downloads', '?'):>8}  "
                  f"weights={weight_bytes / 1e9:5.1f}GB  dtype={dtype}  gated={getattr(info, 'gated', None)}")
            shown += 1
            if shown >= args.show:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"{m.id:60s} skipped ({type(exc).__name__})")


if __name__ == "__main__":
    main()
