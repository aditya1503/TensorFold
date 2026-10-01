"""Unit tests for MTP with concurrency - logic tests without full model."""

import pytest
import torch
from dataclasses import dataclass, field
from typing import Any, List

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def test_lane_mtp_fields():
    """Test Lane dataclass has MTP fields."""
    from tensorfold.families.glm5_next.cuda.multi import Lane, Stream
    
    s = Stream(prompt=[1, 2, 3], count=10)
    
    lane = Lane(
        s=s,
        sid=0,
        slot=0,
        extent=None,
        st=None,
        order=0,
        code=[3, 5, 300000, 0],
        spec="c5:0.3",
        policy=None,
        dflash=False,
    )
    
    # Check MTP fields exist
    assert hasattr(lane, 'mtp')
    assert hasattr(lane, 'last_hidden')
    assert hasattr(lane, 'pending_tokens')
    
    # Check defaults
    assert lane.mtp is False
    assert lane.last_hidden is None
    assert lane.pending_tokens == []
    
    # Check we can set them
    lane.mtp = True
    lane.last_hidden = torch.zeros(1, 512)
    lane.pending_tokens = [42]
    
    assert lane.mtp is True
    assert lane.last_hidden is not None
    assert lane.pending_tokens == [42]


def test_mtp_policy_remapping():
    """Test that MTP policies are remapped to DFlash2 in multi-stream mode."""
    from tensorfold.families.glm5_next.cuda.multi import multi_code
    
    dflash_policy = [11, 5, 300000, 0]  # DFlash2 version of c5:0.3
    
    # MTP policy (kind 3) should become DFlash2
    mtp_policy = [3, 5, 300000, 0]
    remapped = multi_code(mtp_policy, dflash=True, dflash_policy=dflash_policy)
    assert remapped == dflash_policy
    
    # Serial policy (kind 0) stays serial
    serial_policy = [0, 0, 0, 0]
    remapped = multi_code(serial_policy, dflash=True, dflash_policy=dflash_policy)
    assert remapped == [0, 0, 0, 0]
    
    # DFlash2 policies (kind 1x) stay as-is
    dflash2_policy = [11, 5, 300000, 0]
    remapped = multi_code(dflash2_policy, dflash=True, dflash_policy=dflash_policy)
    assert remapped == dflash2_policy


def test_multi_prefill_piece_mtp_field():
    """Test Piece dataclass has MTP field."""
    from tensorfold.families.glm5_next.cuda.multi_prefill import Piece
    from tensorfold.families.glm5_next.cuda.forward import State
    
    st = State.__new__(State)  # Create without init
    st.pos = 0
    
    piece = Piece(
        st=st,
        tokens=[1, 2, 3],
        drafter=None,
        head=True,
    )
    
    assert hasattr(piece, 'mtp')
    assert piece.mtp is False
    
    piece.mtp = True
    assert piece.mtp is True


def test_draft_logic_mtp_path():
    """Test the MTP draft logic in multi.py round()."""
    from tensorfold.families.glm5_next.cuda.multi import Lane, MAX_WINDOW
    
    # Simulate a lane with MTP enabled
    lane = Lane(
        s=None,
        sid=0,
        slot=0,
        extent=None,
        st=None,
        order=0,
        code=[3, 5, 300000, 0],
        policy=None,  # would be DepthPolicy
        dflash=False,
        mtp=True,
        last_hidden=torch.zeros(1, 512),
        pending_tokens=[5],
        depth=3,
    )
    
    # Verify MTP lane is recognized
    assert lane.mtp is True
    assert lane.last_hidden is not None
    assert lane.pending_tokens == [5]
    assert lane.depth > 0


def test_max_window_constant():
    """Test MAX_WINDOW is defined correctly."""
    from tensorfold.families.glm5_next.cuda.multi import MAX_WINDOW
    
    assert MAX_WINDOW == 32


def test_trim_function():
    """Test the trim function for capping window sizes."""
    from tensorfold.families.glm5_next.cuda.multi import trim, MAX_WINDOW
    
    # Simple case - under cap
    drafts = [[1, 2], [3], [4, 5, 6]]
    result = trim(drafts, MAX_WINDOW)
    assert result == drafts
    
    # Over cap - should trim longest first
    drafts = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14]]
    # Total: 8 + 4 + 2 = 14, plus 3 pending = 17 > 32? No, 17 < 32
    result = trim(drafts, 10)  # Use smaller cap for test
    # 8 + 4 + 2 + 3 pending = 17 > 10
    # Should trim from longest (first)
    total = sum(1 + len(d) for d in result)
    assert total <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])