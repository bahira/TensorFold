"""Tiny SwiGLU / GPT-2 on GPU (or CPU torch): cuBLAS + SuperSpear + TensorFold drafts.

On NVIDIA: fp16 weights, Triton champions when Triton is there. Without CUDA this still
runs on CPU torch so the generate/draft contract can be tested on a box with no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tensorfold.drafters.draft_ngram import SessionNGram
from tensorfold.engine.cpu_spear import ngram_propose
from tensorfold.engine.exact_sampling import Sampling, choose, seed_for


def _torch():
    import torch

    return torch


def _device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class GPUConfig:
    kind: str
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    n_inner: int
    n_positions: int
    eps: float = 1e-5
    rope_theta: float = 10000.0
    dtype: str = "float16"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class GPUModel:
    def __init__(self, cfg: GPUConfig, *, act: str, seed: int = 0, device=None):
        torch = _torch()
        self.cfg = cfg
        self.act = act
        self.device = device or _device()
        if self.device.type == "cpu" and cfg.dtype == "float16":
            self.dt = torch.float32          # CPU fp16 matmul is slow and poorly supported
        else:
            self.dt = getattr(torch, cfg.dtype)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        scale = 0.02

        def p(*shape):
            return (torch.randn(*shape, generator=g) * scale).to(device=self.device, dtype=self.dt)

        def ones(n):
            return torch.ones(n, device=self.device, dtype=self.dt)

        def zeros(n):
            return torch.zeros(n, device=self.device, dtype=self.dt)

        self.wte = p(cfg.vocab_size, cfg.n_embd)
        self.wpe = p(cfg.n_positions, cfg.n_embd) if cfg.kind == "gpt2" else None
        self.blocks = []
        for _ in range(cfg.n_layer):
            if cfg.kind == "gpt2":
                self.blocks.append({
                    "ln1_w": ones(cfg.n_embd), "ln1_b": zeros(cfg.n_embd),
                    "ln2_w": ones(cfg.n_embd), "ln2_b": zeros(cfg.n_embd),
                    "c_attn": p(3 * cfg.n_embd, cfg.n_embd), "c_attn_b": p(3 * cfg.n_embd),
                    "c_proj": p(cfg.n_embd, cfg.n_embd),
                    "c_fc": p(cfg.n_inner, cfg.n_embd), "c_fc_b": p(cfg.n_inner),
                    "c_down": p(cfg.n_embd, cfg.n_inner),
                })
            else:
                self.blocks.append({
                    "rms1": ones(cfg.n_embd), "rms2": ones(cfg.n_embd),
                    "Wq": p(cfg.n_embd, cfg.n_embd), "Wk": p(cfg.n_embd, cfg.n_embd),
                    "Wv": p(cfg.n_embd, cfg.n_embd), "Wo": p(cfg.n_embd, cfg.n_embd),
                    "Wgate": p(cfg.n_inner, cfg.n_embd), "Wup": p(cfg.n_inner, cfg.n_embd),
                    "Wdown": p(cfg.n_embd, cfg.n_inner),
                })
        self.ln_w = ones(cfg.n_embd)
        self.ln_b = zeros(cfg.n_embd) if cfg.kind == "gpt2" else None
        self.max_seq = cfg.n_positions
        self.k = torch.zeros(cfg.n_layer, self.max_seq, cfg.n_head, cfg.head_dim,
                             device=self.device, dtype=self.dt)
        self.v = torch.zeros_like(self.k)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def _ln(self, x, w, b):
        torch = _torch()
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        return (x - mu) * torch.rsqrt(var + self.cfg.eps) * w + b

    def _rms(self, x, w):
        torch = _torch()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.eps) * w

    def _rope(self, x, pos0):
        torch = _torch()
        T, _, D = x.shape
        half = D // 2
        freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        t = (pos0 + torch.arange(T, device=x.device, dtype=torch.float32))[:, None]
        ang = t * freq[None, :]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        a, b = x[..., :half], x[..., half:]
        return torch.cat((a * cos[:, None, :] - b * sin[:, None, :],
                          b * cos[:, None, :] + a * sin[:, None, :]), dim=-1)

    def _attn(self, q, k, v):
        torch = _torch()
        scale = q.shape[-1] ** -0.5
        scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
        tq, tk = q.shape[0], k.shape[0]
        if tq != 1:
            mask = torch.triu(torch.ones(tq, tk, dtype=torch.bool, device=q.device), diagonal=tk - tq + 1)
            scores = scores.masked_fill(mask[None], -1e4)
        p = torch.softmax(scores, dim=-1).to(v.dtype)
        return torch.einsum("hts,shd->thd", p, v)

    def _gelu(self, x):
        from tensorfold.kernels.spear.v1 import cuda as spear_cuda

        return spear_cuda.act({"exact": "gelu_exact", "alg": "gelu_alg", "fast": "gelu_fast"}[self.act], x)

    def _swiglu(self, gate, up):
        from tensorfold.kernels.spear.v1 import cuda as spear_cuda

        slot = {"exact": "silu_exact", "alg": "silu_alg", "fast": "silu_fast"}[self.act]
        return spear_cuda.swiglu(gate, up, silu=slot)

    def forward(self, ids: Sequence[int]):
        torch = _torch()
        F = torch.nn.functional
        ids_t = torch.tensor(list(ids), device=self.device, dtype=torch.long)
        x = self.wte[ids_t]
        if self.wpe is not None:
            x = x + self.wpe[self.pos:self.pos + ids_t.numel()]
        T, D = x.shape
        H, hd = self.cfg.n_head, self.cfg.head_dim
        pos0 = self.pos
        for i, b in enumerate(self.blocks):
            if self.cfg.kind == "gpt2":
                h = self._ln(x, b["ln1_w"], b["ln1_b"])
                qkv = F.linear(h, b["c_attn"], b["c_attn_b"]).view(T, 3, H, hd)
                q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
                self.k[i, pos0:pos0 + T], self.v[i, pos0:pos0 + T] = k, v
                attn = self._attn(q, self.k[i, :pos0 + T], self.v[i, :pos0 + T]).reshape(T, D)
                x = x + F.linear(attn, b["c_proj"])
                n = self._ln(x, b["ln2_w"], b["ln2_b"])
                x = x + F.linear(self._gelu(F.linear(n, b["c_fc"], b["c_fc_b"])), b["c_down"])
            else:
                h = self._rms(x, b["rms1"])
                q = F.linear(h, b["Wq"]).view(T, H, hd)
                k = F.linear(h, b["Wk"]).view(T, H, hd)
                v = F.linear(h, b["Wv"]).view(T, H, hd)
                q, k = self._rope(q, pos0), self._rope(k, pos0)
                self.k[i, pos0:pos0 + T], self.v[i, pos0:pos0 + T] = k, v
                attn = self._attn(q, self.k[i, :pos0 + T], self.v[i, :pos0 + T]).reshape(T, D)
                x = x + F.linear(attn, b["Wo"])
                n = self._rms(x, b["rms2"])
                x = x + F.linear(self._swiglu(F.linear(n, b["Wgate"]), F.linear(n, b["Wup"])), b["Wdown"])
        self.pos += int(T)
        last = x[-1]
        if self.cfg.kind == "gpt2":
            last = self._ln(last, self.ln_w, self.ln_b)
        else:
            last = self._rms(last, self.ln_w)
        return (last @ self.wte.T).float()

    def sample(self, logits, position: int, sampling: Sampling) -> int:
        logits_np = logits.detach().float().cpu().numpy()
        if sampling.temperature <= 0:
            return int(np.argmax(logits_np))
        vocab = logits_np.shape[-1]
        k = min(sampling.top_k if sampling.top_k else 64, vocab)
        idx = np.argpartition(-logits_np, kth=k - 1)[:k]
        return choose(logits_np[idx], idx.astype(np.int64), position, sampling)


def tiny_gpu(*, kind: str = "swiglu", act: str = "alg", n_layer: int = 2, n_embd: int = 128,
             n_head: int = 4, n_inner: int = 256, vocab: int = 256, seq: int = 256,
             seed: int = 0, dtype: str = "float16", device=None) -> GPUModel:
    cfg = GPUConfig(kind, vocab, n_embd, n_head, n_layer, n_inner, seq, dtype=dtype)
    return GPUModel(cfg, act=act, seed=seed, device=device)


def generate(model: GPUModel, prompt: Sequence[int], n_new: int, *,
             sampling: Sampling | None = None, drafts: int = 0,
             ngram: SessionNGram | None = None) -> dict[str, Any]:
    torch = _torch()
    sampling = sampling or Sampling(seed=seed_for(prompt), temperature=0.0)
    prompt_ids = [int(t) for t in prompt]
    ids = list(prompt_ids)
    model.reset()
    with torch.no_grad():
        logits = model.forward(ids)
    if ngram is not None:
        ngram.reset()
        ngram.update(ids)
    committed = proposed = accepted = rounds = 0
    while committed < n_new:
        rounds += 1
        token = model.sample(logits, len(ids), sampling)
        ids.append(token)
        committed += 1
        if ngram is not None:
            ngram.update(ids)
        if committed >= n_new:
            break
        guess = ngram_propose(ngram, ids, min(int(drafts), n_new - committed)) if drafts and ngram else []
        proposed += len(guess)
        with torch.no_grad():
            logits = model.forward([token])
        cur = token
        for g in guess:
            nxt = model.sample(logits, len(ids), sampling)
            if nxt != g:
                break
            ids.append(nxt)
            accepted += 1
            committed += 1
            cur = nxt
            if ngram is not None:
                ngram.update(ids)
            if committed >= n_new:
                break
            with torch.no_grad():
                logits = model.forward([cur])
    _sync()
    return {"ids": ids, "new": ids[len(prompt_ids):], "rounds": rounds,
            "proposed": proposed, "accepted": accepted, "tokens": committed}
