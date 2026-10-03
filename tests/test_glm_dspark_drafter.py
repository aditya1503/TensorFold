"""The RedHatAI DSpark drafter (speculators format) — everything that runs without a GPU: format sniffing, config
reading, the tiny-checkpoint loader, the Markov/confidence chain, `propose`'s all-rows depth, and the concurrent
drafter's pure pieces. The GPU-only cases — the block pass's parity against a reference forward, verify acceptance
and --parallel timing with the real weights — are the weight test (docs/recipes/glm-5.3-flash.md).

Test plan, in order:
1. `drafter_kind` / `drafter_class` / `drafter_taps`: both formats sniff, anything else is refused, and the two
   ranks' parity input (the tap count) reads both layouts.
2. `read_speculators_config`: the RedHatAI fields map (block, mask, six tap layers, backbone dims, Markov and
   confidence dims), and the unsupported variants (own-position drafting, non-causal blocks, gated/rnn heads,
   plain DFlash, a non-qwen3 backbone) are refused with a reason.
3. The loader: a tiny synthetic safetensors checkpoint maps every tensor to its slot, splits by rank, sizes the
   candidate pack and hidden rows, and refuses a vocabulary mismatch with the target.
4. `chain`: the low-rank Markov bias applies in pick order (the anchor first, then each pick), the learned
   confidence head stops a chain below its cumulative threshold, `confs` collection walks whole, and the
   noise-aware cut (`beta` with `round_ms`) is DFlash2's.
5. `propose`: `sample_from_anchor` drafts from every block row (the cap is the block, not one less), an empty
   context drafts nothing, and the chain sees the pending token as its anchor.
6. The concurrent pieces: `live_streams` (which streams draft a round, depths clamped) and `tap_pieces` (several
   streams' taps split into launches, positions in order); `make_multi` picks the right multi drafter.
7. The capacity estimators read both formats: the drafter's geometry and weight estimate (which must not count
   the checkpoint's copies of the target's embedding and head, and must count the Markov tables' fp32 host ride).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def dspark_config(**over):
    """RedHatAI's config.json at tiny dims (D 64, block 4, vocab 128, three tap layers)."""

    cfg = {
        "architectures": ["DSparkDraftModel"], "speculators_model_type": "dspark",
        "block_size": 4, "mask_token_id": 126, "draft_vocab_size": 128,
        "sample_from_anchor": True, "sliding_window_non_causal": False,
        "markov_head_type": "vanilla", "markov_rank": 8,
        "enable_confidence_head": True, "confidence_head_with_markov": True,
        "aux_hidden_state_layer_ids": [1, 2, 3],
        "transformer_layer_config": {
            "model_type": "qwen3", "hidden_size": 64, "head_dim": 16,
            "num_attention_heads": 8, "num_key_value_heads": 8, "num_hidden_layers": 2,
            "intermediate_size": 128, "rms_norm_eps": 1e-05, "sliding_window": 32,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "vocab_size": 128, "max_position_embeddings": 4096,
        },
    }
    cfg.update(over)
    return cfg


def tiny_checkpoint(tmp_path, cfg=None):
    """A speculators DSpark checkpoint with random weights at the config's tiny dims."""

    cfg = dspark_config() if cfg is None else cfg
    d = Path(tmp_path) / "dspark"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    tlc = cfg["transformer_layer_config"]
    V, D, R = 128, int(tlc["hidden_size"]), int(cfg["markov_rank"])
    H, KV, hd, I = (int(tlc[k]) for k in ("num_attention_heads", "num_key_value_heads", "head_dim",
                                          "intermediate_size"))
    w = {
        "embed_tokens.weight": torch.randn(V, D), "lm_head.weight": torch.randn(V, D),
        "fc.weight": torch.randn(D, len(cfg["aux_hidden_state_layer_ids"]) * D),
        "hidden_norm.weight": torch.ones(D), "norm.weight": torch.ones(D),
        "markov_head.markov_w1.weight": torch.randn(V, R), "markov_head.markov_w2.weight": torch.randn(V, R),
        "confidence_head.proj.weight": torch.randn(1, D + R), "confidence_head.proj.bias": torch.randn(1),
    }
    for i in range(int(tlc["num_hidden_layers"])):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = torch.ones(D)
        w[p + "post_attention_layernorm.weight"] = torch.ones(D)
        w[p + "self_attn.q_proj.weight"] = torch.randn(H * hd, D)
        w[p + "self_attn.k_proj.weight"] = torch.randn(KV * hd, D)
        w[p + "self_attn.v_proj.weight"] = torch.randn(KV * hd, D)
        w[p + "self_attn.o_proj.weight"] = torch.randn(D, H * hd)
        w[p + "self_attn.q_norm.weight"] = torch.ones(hd)
        w[p + "self_attn.k_norm.weight"] = torch.ones(hd)
        w[p + "mlp.gate_proj.weight"] = torch.randn(I, D)
        w[p + "mlp.up_proj.weight"] = torch.randn(I, D)
        w[p + "mlp.down_proj.weight"] = torch.randn(D, I)
    save_file(w, str(d / "model.safetensors"))
    return d


def dflash2_dir(tmp_path):
    """An incoai DFlash2 folder: only its config.json matters to the sniffing tests."""

    d = Path(tmp_path) / "dflash2"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"hidden_size": 64, "dflash_config": {
        "block_size": 16, "mask_token_id": 9, "target_layer_ids": [7, 8, 9, 10, 11]}}))
    return d


def fake_weights(vocab=128, world=1):
    return SimpleNamespace(device=torch.device("cpu"), rank=0, world=world, comm=None,
                           cfg=SimpleNamespace(vocab=vocab), embed=None, head=None, draft_head=None)


def load_module():
    import importlib

    return importlib.import_module("tensorfold.families.glm5_next.cuda.dspark")


# -- 1. the format sniffs and the ranks' parity input ------------------------------------------------------------
def test_kind_sniffs_both_formats_and_refuses_the_rest(tmp_path, allocations):  # noqa: F811
    mod = load_module()
    assert mod.drafter_kind(tiny_checkpoint(tmp_path)) == "dspark"
    assert mod.drafter_kind(dflash2_dir(tmp_path)) == "dflash2"
    assert mod.drafter_class(tiny_checkpoint(tmp_path)) is mod.DSparkDrafter
    bad = Path(tmp_path) / "bad"
    bad.mkdir()
    (bad / "config.json").write_text(json.dumps({"architectures": ["SomeOtherModel"]}))
    with pytest.raises(ValueError, match="neither a DFlash2"):
        mod.drafter_kind(bad)


def test_tap_counts_read_both_layouts_for_rank_parity(tmp_path, allocations):  # noqa: F811
    mod = load_module()
    assert mod.drafter_taps(None) == 0
    assert mod.drafter_taps(tiny_checkpoint(tmp_path)) == 3
    assert mod.drafter_taps(dflash2_dir(tmp_path)) == 5


# -- 2. the config maps, and the unsupported variants are refused -----------------------------------------------
def test_config_maps_the_speculators_fields(allocations):  # noqa: F811
    mod = load_module()
    c = mod.read_speculators_config(dspark_config())
    assert (c["block"], c["mask_id"], c["taps"]) == (4, 126, (1, 2, 3))
    assert (c["hidden"], c["hd"], c["heads"], c["kv_heads"], c["inter"], c["layers"]) == (64, 16, 8, 8, 128, 2)
    assert (c["window"], c["causal"], c["vocab"]) == (31, True, 128)
    assert (c["markov_rank"], c["confidence"], c["anchor_drafts"]) == (8, True, True)
    assert c["eps"] == 1e-05 and c["theta"] == 10000.0


@pytest.mark.parametrize("over,match", [
    ({"sample_from_anchor": False}, "anchor"),
    ({"sliding_window_non_causal": True}, "causally"),
    ({"markov_head_type": "rnn"}, "vanilla"),
    ({"markov_rank": 0}, "markov_rank"),
    ({"transformer_layer_config": {"model_type": "llama"}}, "qwen3"),
])
def test_config_refuses_what_the_port_does_not_run(over, match, allocations):  # noqa: F811
    mod = load_module()
    with pytest.raises(ValueError, match=match):
        mod.read_speculators_config(dspark_config(**over))


# -- 3. the loader maps a tiny checkpoint ----------------------------------------------------------------------
def test_loader_maps_tensors_and_sizes_buffers(tmp_path, allocations, monkeypatch):  # noqa: F811
    mod = load_module()
    monkeypatch.delenv("TF_GLM_DSPARK_TOP_K", raising=False)
    monkeypatch.delenv("TF_GLM_DRAFT_QUANT", raising=False)
    d = mod.DSparkDrafter(tiny_checkpoint(tmp_path), fake_weights(), capacity=64)
    assert (d.block, d.mask_id, d.tap_layers) == (4, 126, (1, 2, 3))
    assert (d.D, d.hd, d.heads, d.kvh, d.inter) == (64, 16, 8, 8, 128)
    assert d.window == 31 and d.causal and d.anchor_drafts
    assert d.top_k == mod.DEFAULT_TOP_K and len(d.layers) == 2
    assert d.markov_w1.shape == (128, 8) and d.markov_w2.dtype == np.float32
    assert d.conf_w.shape == (64 + 8,) and isinstance(d.conf_b, float)
    assert torch.equal(d.ids, torch.full((4,), 126, dtype=torch.int32))
    assert len(d.kc) == 2 and tuple(d.kc[0].shape) == (8, 64 + 4, 16)      # a flat buffer, capacity + block
    assert tuple(d.tap_in.shape) == (64, 3 * 64)
    assert d.cand_n == 4 * 2 * mod.DEFAULT_TOP_K                           # gathered 1: every block row's top-k
    assert tuple(d.hidden.shape) == (4, 64)                                # the confidence head's input rows
    assert d.nbytes() > 0


def test_loader_refuses_a_vocabulary_mismatch(tmp_path, allocations):  # noqa: F811
    mod = load_module()
    with pytest.raises(ValueError, match="vocabulary"):
        mod.DSparkDrafter(tiny_checkpoint(tmp_path), fake_weights(vocab=151552))


def test_loader_splits_a_second_rank(tmp_path, allocations):  # noqa: F811
    mod = load_module()
    d = mod.DSparkDrafter(tiny_checkpoint(tmp_path), fake_weights(world=2), capacity=64)
    assert (d.heads, d.kvh, d.inter) == (4, 4, 64) and d.gathered == 1      # no comm in tests: no candidate gather


# -- 4. the chain: the Markov bias, the confidence stop, the noise-aware cut ------------------------------------
def chain_drafter(mod):
    """A bare drafter with hand-set tables: W1, W2 orthogonal so each pick's bias names its prev."""

    d = mod.DSparkDrafter.__new__(mod.DSparkDrafter)
    d.markov_rank, d.top_k, d.block = 2, 3, 4
    d.markov_w1 = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    d.markov_w2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    d.conf_w = np.zeros(64 + 2, dtype=np.float32)
    d.conf_w[0] = -10.0                                    # confidence rides hidden[0]
    d.conf_b = 0.0
    return d


def chain_rows(depth=2):
    """tokens [depth, 3] over a 3-token vocabulary, flat logits, hidden rows with the confidence signal first."""

    tokens = np.tile(np.array([[0, 1, 2]], dtype=np.int64), (depth, 1))
    values = np.ones((depth, 3))
    hidden = np.zeros((depth, 64))
    hidden[0, 0] = -1.0                                    # row 0: sigmoid(10) ~ 1
    hidden[1:, 0] = 1.0                                    # the rest: sigmoid(-10) ~ 5e-5
    return tokens, values, hidden


def test_chain_bias_follows_the_picks_in_order(allocations):  # noqa: F811
    mod = load_module()
    d = chain_drafter(mod)
    tokens, values, _ = chain_rows(2)
    # row 0, prev the anchor 2: biases (0.5, 0.5, 1.0) over (1, 1, 1) -> pick 2; row 1, prev 2 again by the tie
    assert d.chain(tokens, values, np.zeros((2, 64)), 2, 1, None) == [2, 2]
    # row 0's biggest value picks 0; row 1 now conditions on 0: biases (0, 1, 1) -> pick 1
    values[0] = np.array([5.0, 1.0, 1.0])
    assert d.chain(tokens, values, np.zeros((2, 64)), 2, 1, None) == [0, 1]
    # row 0's biggest value picks 1; row 1 conditions on 1: biases (1, 0, 1) -> the tie picks 0
    values[0] = np.array([1.0, 5.0, 1.0])
    assert d.chain(tokens, values, np.zeros((2, 64)), 2, 1, None) == [1, 0]


def test_chain_stops_below_the_cumulative_confidence(allocations):  # noqa: F811
    mod = load_module()
    d = chain_drafter(mod)
    tokens, values, hidden = chain_rows(4)
    out = d.chain(tokens, values, hidden, 0, 1, None, confidence=0.5)
    assert len(out) == 1                                                  # the first draft always stays; the
    # second row's learned confidence (~5e-5) sinks the product below 0.5


def test_chain_walks_whole_collecting_confs(allocations):  # noqa: F811
    mod = load_module()
    d = chain_drafter(mod)
    tokens, values, hidden = chain_rows(4)
    confs: list = []
    out = d.chain(tokens, values, hidden, 0, 1, None, 0.9, confs=confs)
    assert len(out) == 4 and len(confs) == 4                              # no stop: the caller cuts instead
    assert confs[0] > 0.99 and confs[1] < 1e-4


def test_chain_cuts_where_tokens_over_ms_is_largest(allocations):  # noqa: F811
    mod = load_module()
    d = chain_drafter(mod)
    tokens, values, _ = chain_rows(2)
    # greedy picks (no sampling): the noisy confidence of a flat row is 1/3 at beta 2; a round of 1 draft wins
    out = d.chain(tokens, values, np.zeros((2, 64)), 0, 1, None, beta=2.0, round_ms=(1.0, 1.0, 100.0, 100.0))
    assert len(out) == 1


# -- 5. propose: all rows draft, the anchor is the pending token ------------------------------------------------
def test_propose_caps_depth_at_the_block_and_anchors_on_pending(allocations, monkeypatch):  # noqa: F811
    mod = load_module()
    d = chain_drafter(mod)
    d.context_end = 10
    seen = []

    def candidates(pending, depth):
        seen.append((pending, depth))
        tokens, values, hidden = chain_rows(depth)
        return tokens[:depth], values[:depth], hidden[:depth]

    d.candidates = candidates                                              # a test seam, not a method
    monkeypatch.delenv("TF_GLM_DSPARK_TOP_K", raising=False)
    # depth 9 over a block of 4: all four rows draft (sample_from_anchor), the anchor row first; the chain
    # conditions on the pending token 1 (biases 1, 0, 1 -> 0; then 0 -> 1; alternating)
    assert d.propose(1, 9, None) == [0, 1, 0, 1]
    assert seen == [(1, 4)]
    assert d.propose(1, 0, None) == [] and seen == [(1, 4)]
    d.context_end = 0
    assert d.propose(1, 4, None) == [] and seen == [(1, 4)]                 # nothing before a first commit


# -- 6. the concurrent drafter's pure pieces --------------------------------------------------------------------
def test_live_streams_clamp_depths_and_skip_empty_contexts(allocations):  # noqa: F811
    mod = load_module()
    reqs = [SimpleNamespace(ctx=SimpleNamespace(context_end=10), depth=9),
            SimpleNamespace(ctx=SimpleNamespace(context_end=0), depth=4),     # no context: no draft
            SimpleNamespace(ctx=SimpleNamespace(context_end=3), depth=0),      # no depth asked
            SimpleNamespace(ctx=SimpleNamespace(context_end=1), depth=2)]
    assert mod.live_streams(reqs, block=4) == [(0, 4), (3, 2)]


def test_tap_pieces_split_launches_in_position_order(allocations):  # noqa: F811
    mod = load_module()
    # one stream past a launch's rows: the rest starts the next launch where this one stopped
    assert mod.tap_pieces([(0, 5, 100)], 4) == [[(0, 0, 4, 100)], [(0, 4, 1, 104)]]
    # two streams pack one launch, the overflow rolls into the next
    assert mod.tap_pieces([(0, 2, 0), (1, 3, 50)], 4) == [[(0, 0, 2, 0), (1, 0, 2, 50)], [(1, 2, 1, 52)]]
    # an exact boundary closes a launch with nothing trailing
    assert mod.tap_pieces([(2, 4, 7)], 4) == [[(2, 0, 4, 7)]]
    assert mod.tap_pieces([], 4) == []


def test_make_multi_picks_the_dspark_pool(tmp_path, allocations, monkeypatch):  # noqa: F811
    mod = load_module()
    solo = mod.DSparkDrafter(tiny_checkpoint(tmp_path), fake_weights(), capacity=64)
    md = mod.make_multi(solo, streams=2)
    assert isinstance(md, mod.DSparkMultiDrafter) and md.streams == 2 and md.block == 4
    assert len(md.contexts) == 2 and md.contexts[1].context_end == 0
    assert tuple(md.contexts[0].kc[0].shape) == (8, 68, 16)                 # one slot's view of the pool
    seen = []

    class FakeMulti:                                                      # the DFlash2 fallback dispatches there

        def __init__(self, d, streams=4, **kw):
            seen.append((d, streams))

    monkeypatch.setattr(mod, "MultiDrafter", FakeMulti)
    other = SimpleNamespace()
    assert mod.make_multi(other, 3) is not None and seen == [(other, 3)]
    with pytest.raises(ValueError, match="at least one stream"):
        mod.make_multi(solo, streams=0)


# -- 7. the capacity estimators read both formats ---------------------------------------------------------------
def test_geometry_reads_the_speculators_layout(allocations):  # noqa: F811
    geom = load_module() and __import__("tensorfold.cuda.geometry", fromlist=["dflash2_geometry"])
    cfg = dspark_config()
    g = geom.dflash2_geometry(cfg, 2, 0, ring=False)
    fixed = 16 * max(64, 4) * (64 + 128) * 4                               # the block pass's scratch
    assert g.bytes_at(64) == fixed + 2 * 2 * 4 * 16 * (64 + 4) * 2          # 2 layers, half the kv heads each


def test_weight_estimate_skips_the_targets_copies_and_counts_the_tables(tmp_path, allocations):  # noqa: F811
    geometry = __import__("tensorfold.cuda.geometry", fromlist=["dflash2_weights"])
    full = geometry.dflash2_weights(tiny_checkpoint(tmp_path), 2)
    trimmed = tiny_checkpoint(tmp_path)                                    # the same checkpoint without the
    w = {}                                                                  # verifier's embedding and head copies
    from safetensors.torch import load_file

    for name, t in load_file(str(trimmed / "model.safetensors")).items():
        if name not in ("embed_tokens.weight", "lm_head.weight"):
            w[name] = t
    save_file(w, str(trimmed / "model.safetensors"))
    assert full.resident == geometry.dflash2_weights(trimmed, 2).resident
    assert full.staging > 0
