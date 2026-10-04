"""RedHatAI's DSpark drafter for GLM-5.3-Flash's CUDA engine: a 5-layer block backbone (DFlash without the
convolutions) plus a low-rank Markov logit-bias head and a confidence head, read from the speculators
checkpoint format (`RedHatAI/GLM-5.3-Flash-speculator.dspark-preview`, MIT — it replaces the incoai DFlash2
checkpoint, whose license is CC BY-NC-ND 4.0).

Semantics (speculators' DSparkDraftModel, arXiv 2607.05147's semi-AR drafting):

- The block is [anchor, mask x (block - 1)] at positions context_end .. context_end + block - 1; the anchor row
  holds the pending token, its own k/v come from the draft layers (its taps arrive only after the round's
  verify commits it), and the context rows' k/v come from the target's taps through `fc` and `hidden_norm`.
- `sample_from_anchor = true`: every row k drafts the token at position context_end + k + 1 — the anchor row
  drafts first — so a round proposes up to `block` tokens, not `block - 1` (DFlash2's own-position convention).
- The Markov head adds a low-rank logit bias B = W1 @ W2 to each row's candidates: row k's bias is
  W2[candidates] @ W1[prev], prev the anchor for row 0 and each earlier row's pick after it — a sequential
  correction over a parallel backbone pass. `markov_head_type` is "vanilla" here; gated and rnn are refused.
- The confidence head reads [final-norm hidden | W1[prev]] per row and predicts that position's acceptance
  probability: the learned signal behind DSpark's confidence-scheduled verification. `chain` multiplies it
  into the cumulative stop rule where DFlash2 used the pick's softmax share, so the f<p> stop policies
  (`DepthPolicy`) drive it unchanged.

Test plan: tests/test_glm_dspark_drafter.py covers the pieces that run without CUDA — format sniffing, config
reading, checkpoint mapping on a tiny synthetic safetensors, the Markov/confidence chain, `propose`'s all-rows
depth, and the concurrent drafter's pure pieces (`live_streams`, `tap_pieces`). GPU-only: the block pass's
parity against a reference forward, verify acceptance and --parallel timing with the real weights
(docs/recipes/glm-5.3-flash.md).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from . import glue
from .dflash2 import NOISE, Drafter, _mm, _quantize4, best_depth, merge_candidates, noisy_confidence
from .dflash2_multi import DraftContext, DraftRequest, MultiDrafter, _dattn_seg_kernel
from .weights import Weights

DEFAULT_TOP_K = 16        # candidates a row proposes to the chain (TF_GLM_DSPARK_TOP_K; DFlash2's selector budget)


def drafter_kind(draft_dir: str | Path) -> str:
    """Which drafter a folder holds: incoai's DFlash2 (``dflash_config``) or a speculators DSpark release."""

    cfg = json.loads((Path(draft_dir) / "config.json").read_text())
    if "dflash_config" in cfg:
        return "dflash2"
    if cfg.get("speculators_model_type") == "dspark" or "DSparkDraftModel" in cfg.get("architectures", []):
        return "dspark"
    raise ValueError(f"{Path(draft_dir)} is neither a DFlash2 draft model (a dflash_config key) nor a DSpark one "
                     f"(speculators_model_type dspark); pull RedHatAI/GLM-5.3-Flash-speculator.dspark-preview or "
                     f"incoai's DFlash2 checkpoint")


def drafter_class(draft_dir: str | Path):
    """The class to construct for a drafter folder (``Drafter`` or ``DSparkDrafter``); both take the same call."""

    from .dflash2 import Drafter as DFlash2

    return DSparkDrafter if drafter_kind(draft_dir) == "dspark" else DFlash2


def drafter_taps(draft_dir: str | Path | None) -> int:
    """How many target layers' taps the drafter reads: 0 with no drafter; both ranks must match."""

    if draft_dir is None:
        return 0
    cfg = json.loads((Path(draft_dir) / "config.json").read_text())
    if "dflash_config" in cfg:
        return len(cfg["dflash_config"].get("target_layer_ids", ()))
    return len(cfg.get("aux_hidden_state_layer_ids", ()))


def read_speculators_config(cfg: dict) -> dict:
    """A speculators DSpark config.json -> the fields the drafter reads, refusing what this port does not run."""

    tlc = cfg.get("transformer_layer_config") or {}
    if not tlc or tlc.get("model_type") != "qwen3":
        raise ValueError("a DSpark draft model here needs a qwen3 transformer_layer_config (RedHatAI's has one)")
    out = {
        "block": int(cfg["block_size"]),
        "mask_id": int(cfg["mask_token_id"]),
        # speculators names HF output_hidden_states indices (the state entering a layer); this engine's taps are the
        # states after a layer runs, so an aux id L feeds the drafter from engine layer L - 1. Verified by A/B: the
        # shift takes periodic-text acceptance from 0.32 to 222 of 224 drafts accepted.
        "taps": tuple(int(i) - 1 for i in cfg["aux_hidden_state_layer_ids"]),
        "hidden": int(tlc["hidden_size"]),
        "hd": int(tlc["head_dim"]),
        "heads": int(tlc["num_attention_heads"]),
        "kv_heads": int(tlc["num_key_value_heads"]),
        "inter": int(tlc["intermediate_size"]),
        "layers": int(tlc["num_hidden_layers"]),
        "eps": float(tlc["rms_norm_eps"]),
        "theta": float((tlc.get("rope_parameters") or {}).get("rope_theta", 10000.0)),
        "window": max(0, int(tlc.get("sliding_window", 0)) - 1),
        "vocab": int(cfg.get("draft_vocab_size") or tlc["vocab_size"]),
        "markov_rank": int(cfg.get("markov_rank", 0)),
        "confidence": bool(cfg.get("enable_confidence_head", False)),
        "confidence_with_markov": bool(cfg.get("confidence_head_with_markov", False)),
        "anchor_drafts": bool(cfg.get("sample_from_anchor", True)),
        "causal": not bool(cfg.get("sliding_window_non_causal", False)),
    }
    if not out["anchor_drafts"]:
        raise ValueError("this port drafts from the anchor row (sample_from_anchor true, DSpark's default); "
                         "a false one is DFlash2's own-position convention")
    if not out["causal"]:
        raise ValueError("this port attends the block causally (sliding_window_non_causal false, RedHatAI's)")
    if cfg.get("markov_head_type", "vanilla") not in ("vanilla", None):
        raise ValueError(f"only the vanilla Markov head is ported, not {cfg.get('markov_head_type')!r}")
    if out["markov_rank"] <= 0:
        raise ValueError("a DSpark draft model needs markov_rank > 0 (0 is plain DFlash)")
    if out["confidence"] and out["confidence_with_markov"] and out["markov_rank"] <= 0:
        raise ValueError("confidence_head_with_markov needs a Markov head")
    return out


@dataclass
class DSparkLayer:
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: object               # qmm.Q4: this rank's [q heads | k heads | v heads]
    kv: object                # qmm.Q4: this rank's [k heads | v heads] (context rows)
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: object                 # qmm.Q4: [D, this rank's heads * head_dim], a row-parallel partial
    gu: object                # qmm.Q4: this rank's [gate | up]
    down: object              # qmm.Q4: [D, this rank's MLP width], a row-parallel partial


class DSparkDrafter(Drafter):
    """Draft one sequence with the speculators DSpark checkpoint: DFlash2's interface, all-rows candidates."""

    def __init__(self, draft_dir: str | Path, w: Weights, *, block: int | None = None, capacity: int = 2560,
                 tap_rows: int = 0, ring: bool = False, top_k: int | None = None) -> None:
        path = Path(draft_dir)
        c = read_speculators_config(json.loads((path / "config.json").read_text()))
        self.w = w
        self.dev = w.device
        self.rank, self.world = w.rank, w.world
        if c["vocab"] != int(w.cfg.vocab):
            raise ValueError(f"the DSpark draft model's vocabulary ({c['vocab']}) is not the target's "
                             f"({int(w.cfg.vocab)}): this port drafts with the target's embedding and head, so it "
                             f"takes full-vocabulary models only")
        self.D, self.hd = c["hidden"], c["hd"]
        heads, kv_heads = c["heads"], c["kv_heads"]
        if heads % self.world or kv_heads % self.world:
            raise ValueError("drafter heads must split evenly over the ranks")
        self.heads, self.kvh = heads // self.world, kv_heads // self.world
        self.eps = c["eps"]
        self.mask_id = c["mask_id"]
        self.block = int(block or c["block"])
        self.tap_layers = c["taps"]
        self.window = c["window"]
        self.causal = c["causal"]
        self.anchor_drafts = c["anchor_drafts"]
        self.markov_rank = c["markov_rank"]
        self.confidence = c["confidence"]
        self.top_k = int(top_k or os.environ.get("TF_GLM_DSPARK_TOP_K", str(DEFAULT_TOP_K)))
        inter = c["inter"]
        if inter % self.world:
            raise ValueError("drafter MLP must split evenly over the ranks")
        self.inter = inter // self.world
        r, H, KV, hd = self.rank, self.heads, self.kvh, self.hd
        dev = self.dev

        def gpu(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev, torch.bfloat16).contiguous()

        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            def get(name: str) -> torch.Tensor:
                return f.get_tensor(name)

            self.fc = _quantize4(gpu(get("fc.weight")))
            self.hidden_norm = gpu(get("hidden_norm.weight"))
            self.norm = gpu(get("norm.weight"))
            if self.confidence:
                conf_w = get("confidence_head.proj.weight")
                want = self.D + (self.markov_rank if c["confidence_with_markov"] else 0)
                if tuple(conf_w.shape) != (1, want):
                    raise ValueError(f"the confidence head reads [hidden | markov] of {want}, this one is "
                                     f"{tuple(conf_w.shape)}")
                self.conf_w = conf_w.float().numpy().ravel().copy()
                self.conf_b = float(get("confidence_head.proj.bias").flatten()[0])
            else:
                self.conf_w, self.conf_b = None, 0.0
            markov = get("markov_head.markov_w1.weight"), get("markov_head.markov_w2.weight")
            if any(tuple(t.shape) != (c["vocab"], self.markov_rank) for t in markov):
                raise ValueError(f"the Markov head is two [vocab, rank] tables, got {[tuple(t.shape) for t in markov]}")
            # host fp32 tables like DFlash2's codebooks: the chain gathers rows of them per pick
            self.markov_w1 = markov[0].float().numpy().copy()
            self.markov_w2 = markov[1].float().numpy().copy()
            self.layers: list[DSparkLayer] = []
            for i in range(c["layers"]):
                p = f"layers.{i}."
                q = get(p + "self_attn.q_proj.weight")[r * H * hd:(r + 1) * H * hd]
                k = get(p + "self_attn.k_proj.weight")[r * KV * hd:(r + 1) * KV * hd]
                v = get(p + "self_attn.v_proj.weight")[r * KV * hd:(r + 1) * KV * hd]
                o = get(p + "self_attn.o_proj.weight")[:, r * H * hd:(r + 1) * H * hd]
                g = get(p + "mlp.gate_proj.weight")[r * self.inter:(r + 1) * self.inter]
                u = get(p + "mlp.up_proj.weight")[r * self.inter:(r + 1) * self.inter]
                dn = get(p + "mlp.down_proj.weight")[:, r * self.inter:(r + 1) * self.inter]
                self.layers.append(DSparkLayer(
                    in_norm=gpu(get(p + "input_layernorm.weight")),
                    post_norm=gpu(get(p + "post_attention_layernorm.weight")),
                    qkv=_quantize4(gpu(torch.cat((q, k, v)))),
                    kv=_quantize4(gpu(torch.cat((k, v)))),
                    q_norm=gpu(get(p + "self_attn.q_norm.weight")),
                    k_norm=gpu(get(p + "self_attn.k_norm.weight")),
                    o=_quantize4(gpu(o)),
                    gu=_quantize4(gpu(torch.cat((g, u)))),
                    down=_quantize4(gpu(dn))))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.inv_freq = 1.0 / c["theta"] ** (torch.arange(hd // 2, device=dev, dtype=torch.float32) * 2 / hd)
        # the context buffers and the candidate pack, exactly DFlash2's layout with all block rows drafting
        from tensorfold.cuda.geometry import draft_ring_rows

        self.capacity = capacity
        rows = draft_ring_rows(self.window, self.block) if ring and self.window >= 0 else 0
        self.ring = rows if 0 < rows < capacity + self.block else 0
        self.cap = self.ring or capacity + self.block
        self.kc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.vc = [torch.zeros((KV, self.cap, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.gathered = self.world if self.world > 1 and w.comm is not None else 1
        self.pos_dev = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.context_end = 0
        n = self.block
        self.ids = torch.full((n,), self.mask_id, dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((1,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.ar = torch.arange(max(64, n), device=dev)
        self.tap_in = torch.zeros((64, len(self.tap_layers) * self.D), dtype=torch.bfloat16, device=dev)
        # a pass's candidates (every rank's [block, 2 * top_k], values then ids as fp32 bits) and its final-norm
        # hidden rows [block, D] fp32 — the confidence head's input — in one device buffer, one pinned copy
        self.cand_n = self.gathered * n * 2 * self.top_k
        cuda = torch.cuda.is_available()
        self.cand = torch.zeros((self.cand_n + n * self.D,), dtype=torch.float32, device=dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=cuda)
        self.cand_ready = torch.cuda.Event() if cuda else None
        self.packed = self.cand[:self.cand_n].view(self.gathered, n, 2 * self.top_k)
        self.hidden = self.cand[self.cand_n:].view(n, self.D)
        self.pool = None
        self.block_graph = None
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.tap_rows = min(max(self.block, int(tap_rows)), self.tap_in.shape[0])

    def nbytes(self) -> int:
        total = sum(q.nbytes() for L in self.layers for q in (L.qkv, L.kv, L.o, L.gu, L.down))
        return total + self.fc.nbytes() + 2 * sum(t.numel() * 2 for t in self.kc)

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """DFlash2's layer without the dynamic convolutions: qwen3's norm -> attn -> add -> norm -> swiglu -> add."""

        from .dflash2 import _dattn_kernel

        L = self.layers[i]
        rows = x.shape[0]
        normed, _ = self._norm(x, L.in_norm)
        q, k, v = self._prep(_mm(normed, L.qkv), L, cos, sin, self.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        out = torch.empty((rows, self.heads * self.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_kernel[(self.kvh,)](q, self.kc[i], self.vc[i], out, self.pos_dev, self.window, self.hd ** -0.5,
                                   N=rows, G=self.heads // self.kvh, NH=self.heads, HD=self.hd, CAP=self.cap, BK=64,
                                   CAUSAL=self.causal, RING=bool(self.ring), num_warps=4)
        x = x + self._row(out, L.o)
        normed, _ = self._norm(x, L.post_norm)
        gu = _mm(normed, L.gu)
        act = torch.empty((rows, self.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, self.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, 1e30)
        return x + self._row(act, L.down, axs)

    def _block_compute(self) -> None:
        """Run [pending, mask x (block - 1)] at the committed length; every row's candidates and hidden rows go
        to the pack buffer (the anchor row drafts first: sample_from_anchor), merged over both ranks."""

        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids, self.w.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        idx = self._slots(n)
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, idx)
        h, hs = self._norm(x, self.norm)
        logits = _mm(h, self.w.draft_head if self.w.draft_head is not None else self.w.head, hs)
        vals, local = torch.topk(logits.float(), self.top_k, dim=-1)
        gids = (local + self.w.vocab_offset).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        if self.gathered > 1:
            self.w.comm.all_gather(packed.view(-1), self.cand[:self.cand_n])
        else:
            self.cand[:self.cand_n].copy_(packed.view(-1))
        self.hidden.copy_(h.float())

    @torch.no_grad()
    def candidates(self, pending: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidate ids [depth, top_k] for the positions after ``pending``, their logits, the final-norm hidden
        rows [depth, D] (the confidence head's input): all block rows draft, so depth runs to the block."""

        depth = min(depth, self.block)
        self.ids_host[0] = pending
        self.ids[:1].copy_(self.ids_host, non_blocking=True)
        if self.block_graph is not None:
            self.block_graph.replay()
        else:
            self._block_compute()
        if self.cand_ready is not None:
            self.cand_host.copy_(self.cand, non_blocking=True)
            self.cand_ready.record()
            self.cand_ready.synchronize()
        else:
            self.cand_host.copy_(self.cand)
        g = self.cand_host[:self.cand_n].view(self.gathered, self.block, 2 * self.top_k)[:, :depth]
        hidden = self.cand_host[self.cand_n:].view(self.block, self.D)[:depth]
        # merge_candidates keeps its third argument whole: the hidden rows ride along as the "projection"
        return merge_candidates(g, hidden, self.top_k)

    def confidence_at(self, hidden: np.ndarray, prev: int) -> float:
        """The confidence head's sigmoid over [hidden | W1[prev]]: that position's acceptance probability."""

        feats = np.concatenate([hidden, self.markov_w1[prev]])
        return float(1.0 / (1.0 + np.exp(-(float(self.conf_w @ feats) + self.conf_b))))

    def chain(self, tokens: np.ndarray, values: np.ndarray, hidden: np.ndarray, anchor: int, first: int,
              sampling: Sampling | None, confidence: float = 0.0, *, beta: float = 0.0,
              round_ms: tuple[float, ...] | None = None, confs: list | None = None) -> list[int]:
        """Pick candidates with the low-rank Markov bias of the previous pick (the anchor first) and the target's
        keyed noise; stop below the cumulative confidence — the learned head's sigmoid where DFlash2 used the
        pick's softmax share — always keeping the first draft. ``beta``/``round_ms``/``confs``: DFlash2's
        noise-aware stop rules, unchanged (``noisy_confidence`` estimates the pick, the head the position)."""

        depth = tokens.shape[0]
        sampled = sampling is not None and sampling.temperature > 0
        temp = float(sampling.temperature) if sampled else 1.0
        noise = None
        if sampled:
            noise = -np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(depth), tokens)))
        noisy = beta > 0
        out: list[int] = []
        conf: list[float] = []
        prev, chain = anchor, 1.0
        for d in range(depth):
            bias = self.markov_w2[tokens[d]].astype(np.float64) @ self.markov_w1[prev].astype(np.float64)
            score = (values[d] + bias) / temp
            pick = score + NOISE * noise[d] if noise is not None else score
            j = int(np.argmax(pick))
            if confs is not None:
                if noisy:
                    confs.append(noisy_confidence(score, noise[d] if noise is not None else None, j, beta))
                else:
                    confs.append(self.confidence_at(hidden[d], prev))
            elif noisy:
                c = noisy_confidence(score, noise[d] if noise is not None else None, j, beta)
                if round_ms is not None:
                    conf.append(c)
                else:
                    chain *= c
                    if d > 0 and chain < confidence:
                        break
            elif confidence > 0:
                chain *= self.confidence_at(hidden[d], prev)
                if d > 0 and chain < confidence:
                    break
            prev = int(tokens[d, j])
            out.append(prev)
        if noisy and round_ms is not None and confs is None:
            out = out[:best_depth(conf, round_ms)]
        return out

    def propose(self, pending: int, depth: int, sampling: Sampling | None, confidence: float = 0.0, *,
                beta: float = 0.0, round_ms: tuple[float, ...] | None = None) -> list[int]:
        """Up to ``depth`` drafts for the positions after the pending token; the anchor row drafts first, so
        the cap is the block, not one less than it (``sample_from_anchor``)."""

        depth = min(depth, self.block)
        if depth < 1 or self.context_end == 0:
            return []
        tokens, values, hidden = self.candidates(pending, depth)
        if beta > 0:
            return self.chain(tokens, values, hidden, pending, self.context_end + 1, sampling, confidence,
                              beta=beta, round_ms=round_ms)
        return self.chain(tokens, values, hidden, pending, self.context_end + 1, sampling, confidence)


def live_streams(reqs: Sequence, block: int) -> list[tuple[int, int]]:
    """Which requests draft a round: (index, depth) with the depth clamped to the block over a non-empty context."""

    out = []
    for i, r in enumerate(reqs):
        depth = min(r.depth, block)
        if depth >= 1 and r.ctx.context_end > 0:
            out.append((i, depth))
    return out


def tap_pieces(items: Sequence[tuple[int, int, int]], t_max: int) -> list[list[tuple[int, int, int, int]]]:
    """Split (slot, rows, position of row 0) into launches of at most t_max rows: each piece
    (slot, start, rows, position of its first row), a stream's rows in position order."""

    batches: list[list[tuple[int, int, int, int]]] = []
    used, piece = 0, []
    for slot, rows, end in items:
        start, pos = 0, end
        while start < rows:
            take = min(rows - start, t_max - used)
            piece.append((slot, start, take, pos))
            start, pos, used = start + take, pos + take, used + take
            if used == t_max:
                batches.append(piece)
                piece, used = [], 0
    if piece:
        batches.append(piece)
    return batches


class DSparkMultiDrafter(MultiDrafter):
    """DSpark for several streams at once: one drafter's weights, a context slot per stream, the block passes of
    several streams in one launch sequence — DFlash2's multi drafter without the convolutions, all rows drafting.
    ``DraftContext`` and ``DraftRequest`` are reused; every stream's candidates are its solo pass's bits."""

    def __init__(self, drafter: DSparkDrafter, streams: int = 4, *, tap_rows: int | None = None) -> None:
        d = self.d = drafter
        if streams < 1:
            raise ValueError("a multi-stream drafter needs at least one stream")
        self.streams = streams
        self.dev = d.dev
        self.block = d.block
        self.cap = d.cap
        n = self.block
        rows = max(64, streams * (tap_rows or d.tap_rows))
        self.t_max = -(-rows // n) * n
        self.trash = self.t_max
        self.slot_rows = streams * self.cap + self.trash
        KV, hd = d.kvh, d.hd
        self.kc = [torch.zeros((KV, self.slot_rows, hd), dtype=torch.bfloat16, device=self.dev) for _ in d.layers]
        self.vc = [torch.zeros((KV, self.slot_rows, hd), dtype=torch.bfloat16, device=self.dev) for _ in d.layers]
        self.pos = torch.zeros((streams + 1,), dtype=torch.int64, device=self.dev)
        self.contexts = None                                  # set below, after the class resolves
        pinned = torch.cuda.is_available()
        self.b_meta = torch.zeros((3 * streams,), dtype=torch.int64, device=self.dev)
        self.b_host = torch.zeros((3 * streams,), dtype=torch.int64, pin_memory=pinned)
        self.ids = torch.full((streams * n,), d.mask_id, dtype=torch.int32, device=self.dev)
        self.ar = torch.arange(max(64, n), device=self.dev)
        T = self.t_max
        self.t_meta = torch.zeros((2 * T + 2 * streams,), dtype=torch.int64, device=self.dev)
        self.t_host = torch.zeros((2 * T + 2 * streams,), dtype=torch.int64, pin_memory=pinned)
        self.t_ready = torch.cuda.Event() if pinned else None
        self.tap_in = torch.zeros((T, d.tap_in.shape[1]), dtype=torch.bfloat16, device=self.dev)
        # every stream's candidates and its hidden rows [block, D] (the confidence head's input), one buffer
        m = n
        self.cand_n = d.gathered * streams * m * 2 * d.top_k
        self.cand = torch.zeros((self.cand_n + streams * m * d.D,), dtype=torch.float32, device=self.dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=pinned)
        self.cand_ready = torch.cuda.Event() if pinned else None
        self.hidden_rows = self.cand[self.cand_n:].view(streams * m, d.D)
        self.pool = None
        self.block_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.buckets = list(range(n, T + 1, n))
        self.contexts = [DraftContext(self, i) for i in range(streams)]

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, idx: torch.Tensor, S: int,
                pos: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """``MultiDrafter._layer`` without the convolutions; the segmented attention reads each block's slot."""

        d = self.d
        L = d.layers[i]
        n = self.block
        rows = x.shape[0]
        normed, _ = d._norm(x, L.in_norm)
        q, k, v = d._prep(_mm(normed, L.qkv), L, cos, sin, d.heads)
        self.kc[i].index_copy_(1, idx, k)
        self.vc[i].index_copy_(1, idx, v)
        out = torch.empty((rows, d.heads * d.hd), dtype=torch.bfloat16, device=self.dev)
        _dattn_seg_kernel[(d.kvh, S)](q, self.kc[i], self.vc[i], out, pos, slot, d.window, d.hd ** -0.5, rows,
                                      self.slot_rows, N=n, G=d.heads // d.kvh, NH=d.heads, HD=d.hd, CAP=self.cap,
                                      BK=64, CAUSAL=d.causal, RING=bool(d.ring), num_warps=4)
        x = x + d._row(out, L.o)
        normed, _ = d._norm(x, L.post_norm)
        gu = _mm(normed, L.gu)
        act = torch.empty((rows, d.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, d.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, 1e30)
        return x + d._row(act, L.down, axs)

    def _block_compute(self, S: int) -> None:
        """S streams' blocks [pending, mask x (block - 1)] at their committed lengths; every row's candidates and
        hidden rows, stream s at rows s * block (the anchor row drafts first: all rows)."""

        d, w = self.d, self.d.w
        n, St = self.block, self.streams
        pend, slot, pos = self.b_meta[:S], self.b_meta[St:St + S], self.b_meta[2 * St:2 * St + S]
        self.ids.view(St, n)[:S, 0].copy_(pend)
        rowpos = (pos[:, None] + self.ar[:n][None, :]).reshape(-1)
        idx = (slot[:, None] * self.cap + self._local(rowpos).view(S, n)).reshape(-1)
        R = S * n
        x = torch.empty((R, d.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids[:R], w.embed, d.D, 1, x)
        cos, sin = self._rotary(rowpos)
        for i in range(len(d.layers)):
            x = self._layer(i, x, cos, sin, idx, S, pos, slot)
        h, hs = d._norm(x, d.norm)
        logits = _mm(h, w.draft_head if w.draft_head is not None else w.head, hs).float()
        k = d.top_k
        packed = torch.empty((R, 2 * k), dtype=torch.float32, device=self.dev)
        for s in range(S):                  # torch.topk may pick kernels by row count: one stream each
            rows = slice(s * n, (s + 1) * n)
            vals, local = torch.topk(logits[rows], k, dim=-1)
            gids = (local + w.vocab_offset).to(torch.int32)
            packed[rows] = torch.cat([vals, gids.view(torch.float32)], dim=1)
            self.hidden_rows[rows].copy_(h[rows].float())
        size = d.gathered * R * 2 * k
        if d.gathered > 1:
            w.comm.all_gather(packed.view(-1), self.cand[:size])
        else:
            self.cand[:size].copy_(packed.view(-1))

    @torch.no_grad()
    def candidates(self, reqs: Sequence[tuple]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(stream, pending, depth) -> each stream's solo ``candidates``: one batched pass, all rows drafting."""

        S, St = len(reqs), self.streams
        if not 1 <= S <= St:
            raise ValueError(f"a block pass takes 1..{St} streams, got {S}")
        if len({c.slot for c, _, _ in reqs}) != S or any(c.owner is not self for c, _, _ in reqs):
            raise ValueError("a block pass takes each of this drafter's streams at most once")
        h = self.b_host
        for s, (c, pending, _) in enumerate(reqs):
            h[s], h[St + s], h[2 * St + s] = int(pending), c.slot, c.context_end
        self.b_meta.copy_(h, non_blocking=True)
        g = self.block_graphs.get(S)
        if g is not None:
            g.replay()
        else:
            self._block_compute(S)
        if self.cand_ready is not None:
            self.cand_host.copy_(self.cand, non_blocking=True)
            self.cand_ready.record()
            self.cand_ready.synchronize()
        else:
            self.cand_host.copy_(self.cand)
        d = self.d
        n, k = self.block, d.top_k
        g_all = self.cand_host[:d.gathered * S * n * 2 * k].view(d.gathered, S * n, 2 * k)
        p_all = self.cand_host[self.cand_n:].view(St * n, d.D)
        out = []
        for s, (_, _, depth) in enumerate(reqs):
            out.append(merge_candidates(g_all[:, s * n:s * n + depth], p_all[s * n:s * n + depth], k))
        return out

    def propose(self, reqs: Sequence[DraftRequest]) -> list[list[int]]:
        """Each stream's ``DSparkDrafter.propose``: streams with a depth and a context share one block pass; the
        others get [] (as a solo drafter runs none for them)."""

        out: list[list[int]] = [[] for _ in reqs]
        live = live_streams(reqs, self.block)
        if not live:
            return out
        got = self.candidates([(reqs[i].ctx, reqs[i].pending, depth) for i, depth in live])
        for (i, _), (tokens, values, hidden) in zip(live, got):
            r = reqs[i]
            first = r.ctx.context_end + 1
            if r.beta > 0:
                out[i] = self.d.chain(tokens, values, hidden, r.pending, first, r.sampling, r.confidence,
                                      beta=r.beta, round_ms=r.round_ms)
            else:
                out[i] = self.d.chain(tokens, values, hidden, r.pending, first, r.sampling, r.confidence)
        return out


def make_multi(drafter, streams: int):
    """The multi-stream drafter over a solo one: DFlash2's or DSpark's by its type."""

    if isinstance(drafter, DSparkDrafter):
        return DSparkMultiDrafter(drafter, streams=streams)
    return MultiDrafter(drafter, streams=streams)
