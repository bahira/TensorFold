"""CPU decode of small GPT-2 / SwiGLU models with SuperSpear kernels and TensorFold drafts.

Prefill uses numpy (OpenBLAS). Decode GEMVs use AVX-512 FMA or VNNI int8. Activations
are SuperSpear champions. Drafts use ``SessionNGram``; a draft is kept only when it
equals the keyed serial sample, so output stays byte-identical to ``drafts=0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from tensorfold.drafters.draft_ngram import SessionNGram, key4
from tensorfold.engine.exact_sampling import Sampling, choose, seed_for
from tensorfold.kernels.spear.v1 import native as spear_native
from tensorfold.kernels.spear import v1 as spear

_GELU = {"exact": "gelu_exact", "alg": "gelu_alg", "fast": "gelu_fast"}
_SILU = {"exact": "silu_exact", "alg": "silu_alg", "fast": "silu_fast"}


@dataclass
class ModelConfig:
    kind: str
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    n_inner: int
    n_positions: int
    eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class Linear:
    W: np.ndarray
    b: np.ndarray | None
    packed: spear_native.PackedI8 | None = None

    @classmethod
    def from_out_in(cls, W: np.ndarray, b: np.ndarray | None, *, quant: bool) -> "Linear":
        W = np.ascontiguousarray(W, dtype=np.float32)
        b = None if b is None else np.ascontiguousarray(b, dtype=np.float32)
        packed = spear.pack_i8(W) if quant and spear.available() else None
        return cls(W=W, b=b, packed=packed)

    def __call__(self, x: np.ndarray, *, act: str | None = None) -> np.ndarray:
        # Decode is a single row; keep it a GEMV (VNNI / AVX) even when the caller passed [1, K].
        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1):
            vec = x.reshape(-1)
            if self.packed is not None:
                y = spear.gemv_i8(self.packed, vec, self.b, act=act)
            else:
                y = spear.gemv_f32(self.W, vec, self.b)
                if act:
                    y = spear.act(act, y)
            return y if x.ndim == 1 else y.reshape(1, -1)
        y = (x @ self.W.T).astype(np.float32, copy=False)
        if self.b is not None:
            y += self.b
        return spear.act(act, y) if act else y


def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) * np.reciprocal(np.sqrt(var + eps)) * w + b).astype(np.float32, copy=False)


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    scale = np.reciprocal(np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps))
    return (x * scale * w).astype(np.float32, copy=False)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / e.sum(axis=-1, keepdims=True)


def _causal_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scale = np.float32(1.0 / np.sqrt(q.shape[-1]))
    scores = np.einsum("thd,shd->hts", q, k, optimize=True) * scale
    tq, tk = q.shape[0], k.shape[0]
    if tq != 1:
        mask = np.triu(np.ones((tq, tk), dtype=np.bool_), k=tk - tq + 1)
        scores[:, mask] = np.float32(-1e9)
    return np.einsum("hts,shd->thd", _softmax(scores), v, optimize=True)


def _rope(x: np.ndarray, pos0: int, theta: float) -> np.ndarray:
    T, _, D = x.shape
    half = D // 2
    freq = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    ang = (pos0 + np.arange(T, dtype=np.float32))[:, None] * freq[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    a, b = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = a * cos[:, None, :] - b * sin[:, None, :]
    out[..., half:] = b * cos[:, None, :] + a * sin[:, None, :]
    return out


@dataclass
class GPT2Block:
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    c_attn: Linear
    c_proj: Linear
    c_fc: Linear
    c_proj_mlp: Linear
    n_head: int
    eps: float

    def __call__(self, x: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, pos: int,
                 gelu: str) -> np.ndarray:
        two = x.ndim == 2
        x = x if two else x[None, :]
        T, D = x.shape
        hd = D // self.n_head
        h = _ln(x, self.ln1_w, self.ln1_b, self.eps)
        qkv = self.c_attn(h).reshape(T, 3, self.n_head, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        k_cache[pos:pos + T] = k
        v_cache[pos:pos + T] = v
        attn = _causal_attn(q, k_cache[:pos + T], v_cache[:pos + T]).reshape(T, D)
        x = x + self.c_proj(attn)
        n = _ln(x, self.ln2_w, self.ln2_b, self.eps)
        if T == 1:
            x = x.copy()
            x[0] = x[0] + self.c_proj_mlp(self.c_fc(n[0], act=gelu))
        else:
            x = x + self.c_proj_mlp(spear.act(gelu, self.c_fc(n)))
        return x if two else x[0]


@dataclass
class SwiGLUBlock:
    rms1: np.ndarray
    rms2: np.ndarray
    Wq: Linear
    Wk: Linear
    Wv: Linear
    Wo: Linear
    Wgate: Linear
    Wup: Linear
    Wdown: Linear
    n_head: int
    eps: float
    rope_theta: float

    def __call__(self, x: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, pos: int,
                 silu: str) -> np.ndarray:
        two = x.ndim == 2
        x = x if two else x[None, :]
        T, D = x.shape
        hd = D // self.n_head
        h = _rms(x, self.rms1, self.eps)
        q = self.Wq(h).reshape(T, self.n_head, hd)
        k = self.Wk(h).reshape(T, self.n_head, hd)
        v = self.Wv(h).reshape(T, self.n_head, hd)
        q, k = _rope(q, pos, self.rope_theta), _rope(k, pos, self.rope_theta)
        k_cache[pos:pos + T] = k
        v_cache[pos:pos + T] = v
        attn = _causal_attn(q, k_cache[:pos + T], v_cache[:pos + T]).reshape(T, D)
        x = x + self.Wo(attn)
        n = _rms(x, self.rms2, self.eps)
        if T == 1:
            x = x.copy()
            x[0] = x[0] + self.Wdown(spear.swiglu(self.Wgate(n[0]), self.Wup(n[0]), silu=silu))
        else:
            x = x + self.Wdown(spear.swiglu(self.Wgate(n), self.Wup(n), silu=silu))
        return x if two else x[0]


@dataclass
class CPUModel:
    config: ModelConfig
    wte: np.ndarray
    wpe: np.ndarray | None
    blocks: list
    ln_w: np.ndarray
    ln_b: np.ndarray | None
    act_slot: str
    max_seq: int
    k: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)
    pos: int = 0

    def __post_init__(self) -> None:
        c = self.config
        self.k = np.zeros((c.n_layer, self.max_seq, c.n_head, c.head_dim), dtype=np.float32)
        self.v = np.zeros_like(self.k)

    def reset(self) -> None:
        self.pos = 0

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        x = self.wte[np.asarray(ids, dtype=np.int64)]
        if self.wpe is not None:
            p = np.arange(self.pos, self.pos + np.asarray(ids).size)
            x = x + self.wpe[p]
        return np.ascontiguousarray(x, dtype=np.float32)

    def _norm_head(self, h: np.ndarray) -> np.ndarray:
        if self.config.kind == "gpt2":
            h = _ln(h, self.ln_w, self.ln_b, self.config.eps)
        else:
            h = _rms(h, self.ln_w, self.config.eps)
        return (h @ self.wte.T).astype(np.float32, copy=False)

    def _slot(self) -> str:
        return (_GELU if self.config.kind == "gpt2" else _SILU)[self.act_slot]

    def forward(self, ids: Sequence[int]) -> np.ndarray:
        """Advance the cache by ``ids`` and return the last-row logits."""

        ids_a = np.asarray(ids, dtype=np.int64).reshape(-1)
        x = self._embed(ids_a)
        slot = self._slot()
        pos0 = self.pos
        for i, block in enumerate(self.blocks):
            x = block(x, self.k[i], self.v[i], pos0, slot)
        self.pos += int(ids_a.size)
        last = x[-1] if x.ndim == 2 else x
        return self._norm_head(last)

    def truncate(self, pos: int) -> None:
        self.pos = int(pos)

    def sample(self, logits: np.ndarray, position: int, sampling: Sampling) -> int:
        if sampling.temperature <= 0:
            return int(np.argmax(logits))
        vocab = int(logits.shape[-1])
        k = min(sampling.top_k if sampling.top_k else 64, vocab)
        idx = np.argpartition(-logits, kth=k - 1)[:k]
        return choose(logits[idx], idx.astype(np.int64), position, sampling)


def ngram_propose(ngram: SessionNGram, history: Sequence[int], k: int) -> list[int]:
    """Greedy next tokens from the session n-gram (highest count, then lowest id)."""

    if k <= 0 or ngram.unigram is None:
        return []
    hist = [int(t) for t in history]
    out: list[int] = []
    for _ in range(k):
        t3, t2, t1 = ([None] * 3 + hist)[-3:]
        token = None
        if t1 is not None and t2 is not None and t3 is not None:
            e = ngram.entry(4, key4(t3, t2, t1))
            if e[0]:
                token = max(e[1].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if token is None and t1 is not None and t2 is not None:
            e = ngram.entry(3, (int(t2) << 18) | int(t1))
            if e[0]:
                token = max(e[1].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if token is None and t1 is not None:
            e = ngram.entry(2, int(t1))
            if e[0]:
                token = max(e[1].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if token is None:
            token = int(np.argmax(ngram.unigram))
        out.append(int(token))
        hist.append(int(token))
    return out


def generate(model: CPUModel, prompt: Sequence[int], n_new: int, *,
             sampling: Sampling | None = None, drafts: int = 0,
             ngram: SessionNGram | None = None) -> dict[str, Any]:
    """Prefill ``prompt``, then sample ``n_new`` tokens. Drafts change speed only."""

    sampling = sampling or Sampling(seed=seed_for(prompt), temperature=0.0)
    prompt_ids = [int(t) for t in prompt]
    ids = list(prompt_ids)
    model.reset()
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
        logits = model.forward([token])
        took = 0
        cur = token
        for g in guess:
            nxt = model.sample(logits, len(ids), sampling)
            if nxt != g:
                break
            ids.append(nxt)
            took += 1
            accepted += 1
            committed += 1
            cur = nxt
            if ngram is not None:
                ngram.update(ids)
            if committed >= n_new:
                break
            logits = model.forward([cur])
        # mismatch or no drafts: ``logits`` already belong to the next serial step
    return {
        "ids": ids,
        "new": ids[len(prompt_ids):],
        "rounds": rounds,
        "proposed": proposed,
        "accepted": accepted,
        "tokens": committed,
    }


def tiny_gpt2(*, seed: int = 0, n_layer: int = 2, n_embd: int = 128, n_head: int = 4,
              n_inner: int = 256, vocab: int = 256, seq: int = 256,
              act: str = "alg", quant: bool = False) -> CPUModel:
    rng = np.random.default_rng(seed)
    cfg = ModelConfig("gpt2", vocab, n_embd, n_head, n_layer, n_inner, seq)
    scale = 0.02

    def lin(out: int, inn: int) -> Linear:
        W = rng.normal(0, scale, (out, inn)).astype(np.float32)
        b = rng.normal(0, scale, (out,)).astype(np.float32)
        return Linear.from_out_in(W, b, quant=quant)

    blocks = []
    for _ in range(n_layer):
        blocks.append(GPT2Block(
            ln1_w=np.ones(n_embd, np.float32), ln1_b=np.zeros(n_embd, np.float32),
            ln2_w=np.ones(n_embd, np.float32), ln2_b=np.zeros(n_embd, np.float32),
            c_attn=lin(3 * n_embd, n_embd), c_proj=lin(n_embd, n_embd),
            c_fc=lin(n_inner, n_embd), c_proj_mlp=lin(n_embd, n_inner),
            n_head=n_head, eps=1e-5,
        ))
    wte = rng.normal(0, scale, (vocab, n_embd)).astype(np.float32)
    wpe = rng.normal(0, scale, (seq, n_embd)).astype(np.float32)
    return CPUModel(cfg, wte, wpe, blocks, np.ones(n_embd, np.float32), np.zeros(n_embd, np.float32),
                    act, seq)


def tiny_swiglu(*, seed: int = 0, n_layer: int = 2, n_embd: int = 128, n_head: int = 4,
                n_inner: int = 256, vocab: int = 256, seq: int = 256,
                act: str = "alg", quant: bool = False) -> CPUModel:
    rng = np.random.default_rng(seed)
    cfg = ModelConfig("swiglu", vocab, n_embd, n_head, n_layer, n_inner, seq)
    scale = 0.02

    def lin(out: int, inn: int) -> Linear:
        W = rng.normal(0, scale, (out, inn)).astype(np.float32)
        return Linear.from_out_in(W, None, quant=quant)

    blocks = []
    for _ in range(n_layer):
        blocks.append(SwiGLUBlock(
            rms1=np.ones(n_embd, np.float32), rms2=np.ones(n_embd, np.float32),
            Wq=lin(n_embd, n_embd), Wk=lin(n_embd, n_embd), Wv=lin(n_embd, n_embd), Wo=lin(n_embd, n_embd),
            Wgate=lin(n_inner, n_embd), Wup=lin(n_inner, n_embd), Wdown=lin(n_embd, n_inner),
            n_head=n_head, eps=1e-5, rope_theta=10000.0,
        ))
    wte = rng.normal(0, scale, (vocab, n_embd)).astype(np.float32)
    return CPUModel(cfg, wte, None, blocks, np.ones(n_embd, np.float32), None, act, seq)


def load_gpt2_safetensors(model_dir: str | Path_like, *, act: str = "alg", quant: bool = False,
                          max_seq: int = 512) -> CPUModel:
    """Load a GPT-2 / DistilGPT2 checkpoint (safetensors, Conv1D layout)."""

    from pathlib import Path
    from safetensors.numpy import load_file

    root = Path(model_dir)
    cfg_j = json_load(root / "config.json")
    n_embd = int(cfg_j["n_embd"])
    n_head = int(cfg_j["n_head"])
    n_layer = int(cfg_j["n_layer"])
    n_inner = int(cfg_j.get("n_inner") or 4 * n_embd)
    vocab = int(cfg_j["vocab_size"])
    n_pos = int(cfg_j.get("n_positions") or 1024)
    tensors = {}
    st = root / "model.safetensors"
    if st.is_file():
        tensors = load_file(str(st))
    else:
        raise FileNotFoundError(f"no model.safetensors in {root}")

    def t(*names: str) -> np.ndarray:
        for n in names:
            if n in tensors:
                return np.ascontiguousarray(tensors[n], dtype=np.float32)
        raise KeyError(names[0])

    def conv(weight_name: str, bias_name: str) -> Linear:
        # GPT-2 Conv1D stores [in, out]; we want [out, in]
        W = t(weight_name).T.copy()
        b = t(bias_name)
        return Linear.from_out_in(W, b, quant=quant)

    blocks = []
    for i in range(n_layer):
        p = f"transformer.h.{i}"
        blocks.append(GPT2Block(
            ln1_w=t(f"{p}.ln_1.weight"), ln1_b=t(f"{p}.ln_1.bias"),
            ln2_w=t(f"{p}.ln_2.weight"), ln2_b=t(f"{p}.ln_2.bias"),
            c_attn=conv(f"{p}.attn.c_attn.weight", f"{p}.attn.c_attn.bias"),
            c_proj=conv(f"{p}.attn.c_proj.weight", f"{p}.attn.c_proj.bias"),
            c_fc=conv(f"{p}.mlp.c_fc.weight", f"{p}.mlp.c_fc.bias"),
            c_proj_mlp=conv(f"{p}.mlp.c_proj.weight", f"{p}.mlp.c_proj.bias"),
            n_head=n_head, eps=float(cfg_j.get("layer_norm_epsilon") or 1e-5),
        ))
    cfg = ModelConfig("gpt2", vocab, n_embd, n_head, n_layer, n_inner, n_pos,
                      eps=float(cfg_j.get("layer_norm_epsilon") or 1e-5))
    wte = t("transformer.wte.weight")
    wpe = t("transformer.wpe.weight")
    return CPUModel(cfg, wte, wpe, blocks, t("transformer.ln_f.weight"), t("transformer.ln_f.bias"),
                    act, min(max_seq, n_pos))


def json_load(path) -> dict:
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text())


Path_like = Any
