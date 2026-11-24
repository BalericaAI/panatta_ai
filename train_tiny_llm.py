#!/usr/bin/env python3
"""
train_tiny_llm.py — train a small GPT (Llama-style) from scratch with MLX.

Design goals:
  - self-contained: one file, no config files
  - safe defaults: tiny model so the first run finishes in minutes, not hours
  - visible progress: prints loss, speed, samples checkpoints
  - hard to screw up: auto-downloads data, validates shapes, saves checkpoints

Run:
  python train_tiny_llm.py                 # tiny smoke run (~minutes)
  python train_tiny_llm.py --size small --iters 20000   # overnight model
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


# ---------------------------------------------------------------- model ---

class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, max_seq):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        hd = d // n_heads
        self.scale = hd ** -0.5
        self.q_proj = nn.Linear(d, n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(n_heads * hd, d, bias=False)
        self.rope = nn.RoPE(hd)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
            cache = (k, v)
        offset = k.shape[2] - L
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        # GQA: repeat kv heads
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, max_seq, hidden):
        super().__init__()
        self.attn = Attention(d, n_heads, n_kv_heads, max_seq)
        self.mlp = MLP(d, hidden)
        self.attn_norm = RMSNorm(d)
        self.mlp_norm = RMSNorm(d)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


def _swiglu_hidden(d, multiple=64):
    # Llama-style: ~8/3 * d, rounded up to a multiple of `multiple`.
    hidden = int(8 * d / 3)
    return multiple * ((hidden + multiple - 1) // multiple)


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, max_seq=256, d=384, n_layers=6,
                 n_heads=6, n_kv_heads=3):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        hidden = _swiglu_hidden(d)
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.blocks = [Block(d, n_heads, n_kv_heads, max_seq, hidden)
                       for _ in range(n_layers)]
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def __call__(self, idx):
        x = self.tok_emb(idx)
        mask = "causal" if x.shape[1] > 1 else None
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)

    def loss(self, idx, targets):
        logits = self(idx)
        B, L, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="mean"
        )

    def nparams(self):
        return sum(v.size for _, v in nn.utils.tree_flatten(self.parameters()))

    def generate(self, idx, max_new=256, temp=0.8, top_k=200):
        for _ in range(max_new):
            ctx = idx[:, -self.max_seq:]
            logits = self(ctx)[:, -1, :]
            if temp <= 0:
                nxt = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                logits = logits * (1.0 / temp)
                if top_k and top_k < logits.shape[-1]:
                    kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
                    logits = mx.where(logits < kth, -float("inf"), logits)
                nxt = mx.random.categorical(logits)[:, None]
            idx = mx.concatenate([idx, nxt], axis=1)
            mx.eval(idx)
        return idx


# ------------------------------------------------------------ tokenizer ---

class ByteTokenizer:
    """Byte-level tokenizer: every UTF-8 byte is a token (vocab = 256)."""

    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(int(t) & 0xFF for t in tokens).decode("utf-8", errors="replace")


# ----------------------------------------------------------------- data ---

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")


def load_dataset(data_path, val_frac=0.1):
    path = Path(data_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading dataset -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text(encoding="utf-8", errors="replace")
    ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    n_val = int(len(ids) * val_frac)
    return ids[:-n_val], ids[-n_val:]


def get_batch(ids, batch, seq, rng):
    ix = rng.integers(0, len(ids) - seq - 1, size=batch)
    x = np.stack([ids[i:i + seq] for i in ix])
    y = np.stack([ids[i + 1:i + seq + 1] for i in ix])
    return mx.array(x), mx.array(y)


# -------------------------------------------------------------- configs ---

CFGS = {
    "tiny":  dict(d=384, n_layers=6,  n_heads=6,  n_kv_heads=3),
    "small": dict(d=768, n_layers=12, n_heads=12, n_kv_heads=4),
}


def save_ckpt(model, out_dir, step):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ckpt_{step:06d}.safetensors"
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(path), flat)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(CFGS), default="tiny")
    p.add_argument("--iters", type=int, default=200,
                   help="training iterations (default: quick smoke run)")
    p.add_argument("--seq", type=int, default=256, help="context length")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_ids, val_ids = load_dataset(args.data)
    print(f"dataset: {len(train_ids):,} train bytes, {len(val_ids):,} val bytes")

    tok = ByteTokenizer()
    model = TinyGPT(tok.vocab_size, max_seq=args.seq, **CFGS[args.size])
    mx.eval(model.parameters())
    print(f"model: size={args.size}  params={model.nparams():,}")

    # cosine schedule with linear warmup
    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.iters - args.warmup)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    opt = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))

    t0 = time.time()
    tokens_seen = 0
    for step in range(1, args.iters + 1):
        opt.learning_rate = lr_at(step)
        x, y = get_batch(train_ids, args.batch, args.seq, rng)
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        tokens_seen += args.batch * args.seq

        if step % args.eval_every == 0 or step == 1 or step == args.iters:
            vx, vy = get_batch(val_ids, args.batch, args.seq, rng)
            vloss = model.loss(vx, vy)
            mx.eval(vloss)
            dt = time.time() - t0
            tps = tokens_seen / dt
            print(f"step {step:5d}/{args.iters} | lr {opt.learning_rate.item():.2e} "
                  f"| train {loss.item():.4f} | val {vloss.item():.4f} "
                  f"| {tps:,.0f} tok/s")

        if step % args.ckpt_every == 0:
            print(f"  saved {save_ckpt(model, args.out, step)}")

    path = save_ckpt(model, args.out, args.iters)
    print(f"done in {time.time() - t0:.1f}s. final checkpoint: {path}")


if __name__ == "__main__":
    main()
