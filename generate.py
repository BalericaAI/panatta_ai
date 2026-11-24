#!/usr/bin/env python3
"""
generate.py — load a checkpoint from train_tiny_llm.py and generate text.

  python generate.py --ckpt checkpoints/ckpt_005000.safetensors --prompt "ROMEO:"
  python generate.py --ckpt checkpoints/ckpt_005000.safetensors --interactive
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Reuse model + tokenizer definitions from the trainer so the architecture
# can never drift apart from what was actually trained.
from train_tiny_llm import TinyGPT, ByteTokenizer


CFGS = {
    "tiny":  dict(d=384, n_layers=6,  n_heads=6,  n_kv_heads=3),
    "small": dict(d=768, n_layers=12, n_heads=12, n_kv_heads=4),
}


def load_model(ckpt: Path, size: str, seq: int):
    tok = ByteTokenizer()
    model = TinyGPT(tok.vocab_size, max_seq=seq, **CFGS[size])
    weights = dict(mx.load(str(ckpt)))
    missing = [k for k in nn.utils.tree_flatten(model.parameters())
               if k[0] not in weights]
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors; "
                         f"did you pick the right --size? first missing: "
                         f"{missing[0][0]}")
    model.load_weights(list(weights.items()))
    return tok, model


def generate(tok, model, prompt, max_new, temp, top_k):
    tokens = mx.array([tok.encode(prompt)])
    out = model.generate(tokens, max_new=max_new, temp=temp, top_k=top_k)
    return tok.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to .safetensors checkpoint")
    p.add_argument("--size", choices=list(CFGS), default="tiny",
                   help="must match the size used during training")
    p.add_argument("--seq", type=int, default=256,
                   help="context length, must match training (default 256)")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    tok, model = load_model(Path(args.ckpt), args.size, args.seq)
    print(f"loaded {args.ckpt} ({model.nparams():,} params)")

    if args.interactive:
        print("type a prompt, empty line to quit. (from-scratch byte models "
              "don't chat — prompts work best as text to continue)")
        while True:
            try:
                prompt = input("\n>>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                break
            print(generate(tok, model, prompt, args.max_new, args.temp,
                           args.top_k))
        return

    print(generate(tok, model, args.prompt, args.max_new, args.temp,
                   args.top_k))


if __name__ == "__main__":
    main()
