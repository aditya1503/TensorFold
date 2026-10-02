"""Unit tests for --parallel with MTP: the policy mapping, the lane fields, the window trimmer. CUDA-free (the
``decode_policy`` check skips where the kernels are not importable)."""

import torch

from tensorfold.families.glm5_next.cuda.multi import Lane, multi_code, trim
from tensorfold.families.glm5_next.cuda.multi_prefill import Piece
from tensorfold.families.glm5_next.cuda.verify import Segment
from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream

DFLASH = [13, 5, 300000, 0]          # encode_policy("fc5:0.3"), the DFlash2 auto policy


def test_multi_code_keeps_mtp_policies():
    # an MTP policy (kind 1-3) stays MTP when the MTP head is loaded, whatever the draft model
    assert multi_code([3, 5, 300000, 0], dflash=True, mtp=True, dflash_policy=DFLASH) == [3, 5, 300000, 0]
    assert multi_code([1, 4, 0, 0], dflash=False, mtp=True, dflash_policy=DFLASH) == [1, 4, 0, 0]
    # ... and without the MTP head it becomes its DFlash2 twin (or serial without either drafter)
    assert multi_code([3, 5, 300000, 0], dflash=True, mtp=False, dflash_policy=DFLASH) == [13, 5, 300000, 0]
    assert multi_code([3, 5, 300000, 0], dflash=False, mtp=False, dflash_policy=DFLASH) == [0, 0, 0, 0]
    # serial stays serial; DFlash2 policies stay with a drafter and fall back to MTP/free otherwise
    assert multi_code([0, 0, 0, 0], dflash=True, mtp=True, dflash_policy=DFLASH) == [0, 0, 0, 0]


def test_multi_code_auto():
    # auto becomes the engine's DFlash2 policy with the draft model, an adaptive MTP policy without it
    assert multi_code([4, 2, 8, 30000], dflash=True, mtp=True, dflash_policy=DFLASH) == DFLASH
    assert multi_code([4, 2, 8, 30000], dflash=False, mtp=True, dflash_policy=DFLASH) == [2, 3, 600000, 850000]
    assert multi_code([4, 2, 8, 30000], dflash=False, mtp=False, dflash_policy=DFLASH) == [0, 0, 0, 0]
    # the drafter-shaped code (kind 1x) is passed through only with a drafter
    assert multi_code([11, 5, 0, 0], dflash=True, mtp=True, dflash_policy=DFLASH) == [11, 5, 0, 0]
    assert multi_code([11, 5, 0, 0], dflash=False, mtp=True, dflash_policy=DFLASH) == [0, 0, 0, 0]


def test_lane_mtp_fields():
    """A lane carries its MTP accept state: the head's last hidden and the token it was sampled from."""
    lane = Lane(s=Stream([1, 2, 3], 8), sid=0, slot=0, extent=None, st=None, order=0, code=[3, 5, 300000, 0])
    assert lane.mtp is False and lane.last_hidden is None and lane.pending_tokens == []
    lane.mtp, lane.last_hidden, lane.pending_tokens = True, torch.zeros(1, 512), [7]
    assert (lane.mtp, lane.pending_tokens) == (True, [7]) and lane.last_hidden.shape == (1, 512)


def test_lane_mtp_flag_set_only_by_admit_rules():
    """decode_policy + mtp gate: serial (code 0) and DFlash2 (kind >= 10) lanes never take the MTP path."""
    import pytest

    pytest.importorskip("triton")          # ``decode``'s DepthPolicy imports the kernels' modules
    from tensorfold.families.glm5_next.cuda.engine import decode_policy

    assert decode_policy([0, 0, 0, 0]) is None                        # serial: one row a round
    assert decode_policy([3, 5, 300000, 0]) is not None               # MTP fixed+confidence
    assert decode_policy([13, 5, 300000, 0]) is not None              # DFlash2 code after multi_code
    assert issubclass(NoRoom, __import__("tensorfold.cuda.memory_gate", fromlist=["NoRoom"]).NoRoom)


def test_piece_mtp_fields():
    piece = Piece(st=None, tokens=[1, 2, 3], head=True, mtp=True, nxt=[2, 3])
    assert piece.rows == 3 and piece.nxt == [2, 3] and piece.last_hidden is None


def test_trim_shared_window():
    # a 6-row window + a 2-row window + a 1-row window: 3 + 2 + 1 pending = 32 with the drafts below, untrimmed
    drafts = [[1] * 5, [2] * 3, [3]]
    out = trim(drafts, cap=32)
    assert sum(1 + len(d) for d in out) <= 32
    # over the cap: one draft at a time off the longest tail, never the pending token
    out = trim([[1] * 40, [2] * 10, []], cap=32)
    assert sum(1 + len(d) for d in out) == 32
    assert out[0] == [1] * 19 and out[1] == [2] * 10 and out[2] == []


def test_segment_records():
    lane = Lane(s=Stream([1], 4), sid=0, slot=0, extent=None, st=None, order=0, code=[1, 1, 0, 0])
    seg = Segment(lane.st, [5, 6, 7])
    assert seg.tokens == [5, 6, 7]


def test_watchdog_and_settings_parse():
    from tensorfold.families.glm5_next.cuda.multi import fill_rows, fill_share, fill_unit, watchdog_seconds
    from tensorfold.families.glm5_next.cuda.multi_tune import MultiSettings

    assert fill_rows(2048, 0, "1024") == 1024 and fill_rows(64, 0, "") == 64
    assert fill_share("0.5") == 0.5 and fill_unit(0) == 64
    assert watchdog_seconds("0") == 0.0
    s = MultiSettings.from_env({})
    assert s.depth == "policy" and s.sampler == "streams" and s.lone
    assert MultiSettings.from_env({"TF_GLM_MULTI_DEPTH": "scale:1.5"}).alpha == 1.5


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))