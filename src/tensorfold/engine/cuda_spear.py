"""Tiny SwiGLU / GPT-2 on CUDA: cuBLAS matmul + SuperSpear activations + TensorFold drafts.

This is the notebook path. Weights stay fp16; the matmul is PyTorch/cuBLAS; SiLU/GELU
go through the Triton champions. Same keyed sampler and n-gram contract as the CPU engine.
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

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_spear needs torch.cuda")
    return torch


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
    def __init__(self, cfg: GPUConfig, *, act: str, seed: int = 0):
        torch = _torch()
        self.cfg = cfg
        self.act = act
        self.dt = getattr(torch, cfg.dtype)
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)
        scale = 0.02
        def p(*shape):
            return (torch.randn(*shape, generator=g, device="cuda") * scale).to(self.dt)

        self.wte = p(cfg.vocab_size, cfg.n_embd)
        self.wpe = p(cfg.n_positions, cfg.n_embd) if cfg.kind == "gpt2" else None
        self.blocks = []
        for _ in range(cfg.n_layer):
            if cfg.kind == "gpt2":
                self.blocks.append({
                    "ln1_w": torch.ones(cfg.n_embd, device="cuda", dtype=self.dt),
                    "ln1_b": torch.zeros(cfg.n_embd, device="cuda", dtype=self.dt),
                    "ln2_w": torch.ones(cfg.n_embd, device="cuda", dtype=self.dt),
                    "ln2_b": torch.zeros(cfg.n_embd, device="cuda", dtype=self.dt),
                    "c_attn": p(3 * cfg.n_embd, cfg.n_embd),
                    "c_attn_b": p(3 * cfg.n_embd),
                    "c_proj": p(cfg.n_embd, cfg.n_embd),
                    "c_fc": p(cfg.n_inner, cfg.n_embd),
                    "c_fc_b": p(cfg.n_inner),
                    "c_down": p(cfg.n_embd, cfg.n_inner),
                })
            else:
                self.blocks.append({
                    "rms1": torch.ones(cfg.n_embd, device="cuda", dtype=self.dt),
                    "rms2": torch.ones(cfg.n_embd, device="cuda", dtype=self.dt),
                    "Wq": p(cfg.n_embd, cfg.n_embd), "Wk": p(cfg.n_embd, cfg.n_embd),
                    "Wv": p(cfg.n_embd, cfg.n_embd), "Wo": p(cfg.n_embd, cfg.n_embd),
                    "Wgate": p(cfg.n_inner, cfg.n_embd), "Wup": p(cfg.n_inner, cfg.n_embd),
                    "Wdown": p(cfg.n_embd, cfg.n_inner),
                })
        self.ln_w = torch.ones(cfg.n_embd, device="cuda", dtype=self.dt)
        self.ln_b = torch.zeros(cfg.n_embd, device="cuda", dtype=self.dt) if cfg.kind == "gpt2" else None
        self.max_seq = cfg.n_positions
        self.k = torch.zeros(cfg.n_layer, self.max_seq, cfg.n_head, cfg.head_dim,
                             device="cuda", dtype=self.dt)
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
        import torch.nn.functional as F
        from tensorfold.kernels.spear.v1 import cuda as spear_cuda

        if self.act == "exact":
            return F.gelu(x)
        slot = {"alg": "gelu_alg", "fast": "gelu_fast"}[self.act]
        return spear_cuda.act(slot, x)

    def _swiglu(self, gate, up):
        import torch.nn.functional as F
        from tensorfold.kernels.spear.v1 import cuda as spear_cuda

        if self.act == "exact":
            return F.silu(gate) * up
        slot = {"alg": "silu_alg", "fast": "silu_fast"}[self.act]
        return spear_cuda.swiglu(gate, up, silu=slot)

    def forward(self, ids: Sequence[int]):
        torch = _torch()
        ids_t = torch.tensor(list(ids), device="cuda", dtype=torch.long)
        x = self.wte[ids_t]
        if self.wpe is not None:
            x = x + self.wpe[self.pos:self.pos + ids_t.numel()]
        T, D = x.shape
        H, hd = self.cfg.n_head, self.cfg.head_dim
        pos0 = self.pos
        for i, b in enumerate(self.blocks):
            if self.cfg.kind == "gpt2":
                h = self._ln(x, b["ln1_w"], b["ln1_b"])
                qkv = torch.nn.functional.linear(h, b["c_attn"], b["c_attn_b"]).view(T, 3, H, hd)
                q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
                self.k[i, pos0:pos0 + T], self.v[i, pos0:pos0 + T] = k, v
                attn = self._attn(q, self.k[i, :pos0 + T], self.v[i, :pos0 + T]).reshape(T, D)
                x = x + torch.nn.functional.linear(attn, b["c_proj"])
                n = self._ln(x, b["ln2_w"], b["ln2_b"])
                x = x + torch.nn.functional.linear(self._gelu(torch.nn.functional.linear(n, b["c_fc"], b["c_fc_b"])),
                                                   b["c_down"])
            else:
                h = self._rms(x, b["rms1"])
                q = torch.nn.functional.linear(h, b["Wq"]).view(T, H, hd)
                k = torch.nn.functional.linear(h, b["Wk"]).view(T, H, hd)
                v = torch.nn.functional.linear(h, b["Wv"]).view(T, H, hd)
                q, k = self._rope(q, pos0), self._rope(k, pos0)
                self.k[i, pos0:pos0 + T], self.v[i, pos0:pos0 + T] = k, v
                attn = self._attn(q, self.k[i, :pos0 + T], self.v[i, :pos0 + T]).reshape(T, D)
                x = x + torch.nn.functional.linear(attn, b["Wo"])
                n = self._rms(x, b["rms2"])
                x = x + torch.nn.functional.linear(
                    self._swiglu(torch.nn.functional.linear(n, b["Wgate"]),
                                 torch.nn.functional.linear(n, b["Wup"])),
                    b["Wdown"])
        self.pos += int(T)
        last = x[-1]
        if self.cfg.kind == "gpt2":
            last = self._ln(last, self.ln_w, self.ln_b)
        else:
            last = self._rms(last, self.ln_w)
        return (last @ self.wte.T).float()

    def sample(self, logits, position: int, sampling: Sampling) -> int:
        torch = _torch()
        logits_np = logits.detach().float().cpu().numpy()
        if sampling.temperature <= 0:
            return int(np.argmax(logits_np))
        vocab = logits_np.shape[-1]
        k = min(sampling.top_k if sampling.top_k else 64, vocab)
        idx = np.argpartition(-logits_np, kth=k - 1)[:k]
        return choose(logits_np[idx], idx.astype(np.int64), position, sampling)


def tiny_gpu(*, kind: str = "swiglu", act: str = "alg", n_layer: int = 2, n_embd: int = 128,
             n_head: int = 4, n_inner: int = 256, vocab: int = 256, seq: int = 256,
             seed: int = 0, dtype: str = "float16") -> GPUModel:
    cfg = GPUConfig(kind, vocab, n_embd, n_head, n_layer, n_inner, seq, dtype=dtype)
    return GPUModel(cfg, act=act, seed=seed)


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
    torch.cuda.synchronize()
    return {"ids": ids, "new": ids[len(prompt_ids):], "rounds": rounds,
            "proposed": proposed, "accepted": accepted, "tokens": committed}
