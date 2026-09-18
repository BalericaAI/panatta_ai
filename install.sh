+#!/usr/bin/env python3
+"""
+train_tiny_llm.py — train a small GPT (Llama-style) from scratch with MLX.
+
+Design goals:
+  - self-contained: one file, no config files
+  - safe defaults: tiny model so the first run finishes in minutes, not hours
+  - visible progress: prints loss, speed, samples checkpoints
+  - hard to screw up: auto-downloads data, validates shapes, saves checkpoints
+
+Run:
+  python train_tiny_llm.py                 # tiny smoke run (~minutes)
+  python train_tiny_llm.py --size small --iters 20000   # overnight model
+"""
+
+import argparse
+import json
+import math
+import os
+import time
+from pathlib import Path
+
+import mlx.core as mx
+import mlx.nn as nn
+import mlx.optimizers as optim
+import numpy as np
+
+
+# ---------------------------------------------------------------- model ---
+
+class RMSNorm(nn.Module):
+    def __init__(self, dims, eps=1e-5):
+        super().__init__()
+        self.weight = mx.ones((dims,))
+        self.eps = eps
+
+    def __call__(self, x):
+        return mx.fast.rms_norm(x, self.weight, self.eps)
+
+
+class Attention(nnx := nn.Module):
+    def __init__(self, d, n_heads, n_kv_heads, max_seq):
+        super().__init__()
+        self.n_heads = n_heads
+        self.n_kv_heads = n_kv_heads
+        hd = d // n_heads
+        self.scale = hd ** -0.5
+        self.q_proj = nn.Linear(d, n_heads * hd, bias=False)
+        self.k_proj = nn.Linear(d, n_kv_heads * hd, bias=False)
+        self.v_proj = nn.Linear(d, n_kv_heads * hd, bias=False)
+        self.o_proj = nn.Linear(n_heads * hd, d, bias=False)
+        self.rope = nn.RoPE(hd)
+
+    def __call__(self, x, mask=None, cache=None):
+        B, L, _ = x.shape
+        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
+        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
+        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
+        if cache is not None:
+            k = mx.concatenate([cache[0], k], axis=2)
+            v = mx.concatenate([cache[1], v], axis=2)
+            cache = (k, v)
+        offset = k.shape[2] - L
+        q = self.rope(q, offset=offset)
+        k = self.rope(k, offset=offset)
+        # GQA: repeat kv heads
+        n_rep = self.n_heads // self.n_kv_heads
+        if n_rep > 1:
+            k = mx.repeat(k, n_rep, axis=1)
+            v = mx.repeat(v, n_rep, axis=1)
+        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
+        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
+        return self.o_proj(out)
+
+
+class MLP(nn.Module):
+    def __init__(self, d, hidden):
+        super().__init__()
