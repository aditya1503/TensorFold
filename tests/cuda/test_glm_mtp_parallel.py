"""--parallel with MTP: concurrent requests decode together, each with the reply it gets alone.

Every comparison: the reference is the serial engine (parallel=1) over the same synthetic checkpoint and the same
sampling, so a request's concurrent reply must equal its solo reply bit for bit. The two ranks live in one test
process (``_TwoCopies`` mirrors both-sided all-gathers), which exercises admission, rounds, slot reuse and the
scheduler without a networked peer. Run on a CUDA host:

    python -m pytest tests/cuda/test_glm_mtp_parallel.py -v --timeout=900
"""

from __future__ import annotations

import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _TwoCopies, _checkpoint, _drafter, _generate  # noqa: E402

PROMPTS = [
    [5, 17, 99, 250, 7, 8],
    [1023, 7, 64, 300, 11, 12, 900, 901, 902],
    [13],
    [8, 8, 9, 2000, 31, 77, 1500, 9, 10, 11],
]
LONG_PROMPT = list(range(7, 7 + 1537))                 # spills the default 1,024-row fill window
SAMPLING = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(7, 0.7, 0, 0.9), Sampling(11, 1.0, 20, 1.0)]


def _engines(tmp_path, *, drafter: bool = False, parallel: int = 4, env: dict | None = None):
    """(serial, parallel) engines over one synthetic checkpoint: a reply's reference is its solo decode."""
    import os

    saved = {}
    for k, v in (env or {}).items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine

        model = tmp_path / "model"
        _checkpoint(model)
        args = dict(rank=0, master="", port=0, comm=_TwoCopies())
        if drafter:
            draft = tmp_path / "dflash2"
            _drafter(draft)
            args["drafter"] = draft
        serial = GlmEngine(model, **args)
        par = GlmEngine(model, **{**args, "parallel": parallel, "comm": _TwoCopies()})
        assert par.scheduler is not None and par.multi is not None
    finally:
        for k, v in saved.items():
            os.environ[k] = v if v is not None else os.environ.pop(k, None)
    return serial, par


def _run_parallel(engine, runs: list[tuple[list[int], int, Sampling | None, str]]) -> list[dict]:
    """One thread a request through --parallel's scheduler; returns each thread's (tokens, stats), in order."""

    out: list[tuple[list[int], dict] | None] = [None] * len(runs)
    threads = []
    for i, (prompt, count, sampling, spec) in enumerate(runs):
        def run(i=i, prompt=prompt, count=count, sampling=sampling, spec=spec):
            got: list[int] = []
            stats = _call(engine, prompt, count, sampling, spec, got)
            out[i] = (got, stats)
        threads.append(threading.Thread(target=run))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not any(t.is_alive() for t in threads), "a concurrent request hung (deadlock)"
    return out  # type: ignore[return-value]


def _call(engine, prompt, count, sampling, spec, out):
    engine.request.policy = spec
    engine.request.stop_eos = False
    return engine.generate(list(prompt), count, sampling, lambda new: out.extend(new) or False, draft=True)


def test_mtp_parallel_equals_serial(tmp_path):
    """Four MTP-policy requests decode together; every reply equals its solo one (bit for bit)."""
    serial, parallel = _engines(tmp_path / "mtp_only")
    policies = ["c5:0.3", "3", "0", "auto"]
    refs = [(_generate(serial, p, s, policy=spec, tokens=16)) for p, s, spec in zip(PROMPTS, SAMPLING, policies)]
    runs = [(p, 16, s, spec) for p, s, spec in zip(PROMPTS, SAMPLING, policies)]
    got = _run_parallel(parallel, runs)
    for i, ((ref_tokens, ref_stats), (con_tokens, con_stats)) in enumerate(zip(refs, got)):
        assert con_tokens == ref_tokens, f"stream {i}: concurrent {con_tokens} != solo {ref_tokens}"
        assert con_stats["sha256"] == ref_stats["sha256"]
        if policies[i] != "0":                                          # a drafting policy beats one row a round
            assert con_stats["tokens_per_round"] > 1.0, (i, con_stats)


def test_dflash2_parallel_equals_serial(tmp_path):
    """With the draft model too: the multi-stream DFlash2 path still equals the solo reply."""
    serial, parallel = _engines(tmp_path / "with_drafter", drafter=True)
    refs = [_generate(serial, p, s, policy="fc5:0.3", tokens=16) for p, s in zip(PROMPTS[:2], SAMPLING[:2])]
    got = _run_parallel(parallel, [(p, 16, s, "fc5:0.3") for p, s in zip(PROMPTS[:2], SAMPLING[:2])])
    for i, ((ref_tokens, _), (con_tokens, _)) in enumerate(zip(refs, got)):
        assert con_tokens == ref_tokens, f"stream {i}: concurrent {con_tokens} != solo {ref_tokens}"


def test_mixed_prefill_and_decode(tmp_path):
    """A long prompt chunks in 256-row fills while a short one decodes beside it; both equal their solo runs."""
    serial, parallel = _engines(tmp_path / "mixed", env={"TF_GLM_FILL_ROWS": "256"})
    ref_long, _ = _generate(serial, LONG_PROMPT, Sampling(3, 0.9, 10, 0.95), policy="3", tokens=8)
    ref_short, _ = _generate(serial, PROMPTS[2], SAMPLING[2], policy="3", tokens=8)
    got = _run_parallel(parallel, [(LONG_PROMPT, 8, Sampling(3, 0.9, 10, 0.95), "3"),
                                   (PROMPTS[2], 8, SAMPLING[2], "3")])
    assert got[0][0] == ref_long and got[1][0] == ref_short


def test_cancelled_stream_frees_its_slot_and_reuses_it(tmp_path):
    """A cancelled stream leaves its slot; a later request takes it; every finished reply matches its solo one."""
    serial, parallel = _engines(tmp_path / "reuse", parallel=2)
    ref0, _ = _generate(serial, PROMPTS[0], SAMPLING[0], policy="c5:0.3", tokens=12)
    ref1, _ = _generate(serial, PROMPTS[1], SAMPLING[1], policy="c5:0.3", tokens=12)

    def cancelled(new, stop_after=[True], out=[]):
        out.extend(new)
        if stop_after[0]:
            stop_after[0] = False
            return True                               # the client left: the stream ends after its next round
        return False

    stats0 = [None]

    def run0():
        parallel.request.policy = "c5:0.3"
        parallel.request.stop_eos = False
        stats0[0] = parallel.generate(list(PROMPTS[0]), 12, SAMPLING[0], cancelled, draft=True)

    out1: list[int] = []

    def run1():
        out1.extend(_generate_parallel(parallel, PROMPTS[1], 12, SAMPLING[1], "c5:0.3"))

    t0, t1 = threading.Thread(target=run0), threading.Thread(target=run1)
    t0.start()
    t1.start()
    t0.join(timeout=300)
    t1.join(timeout=300)
    assert not t0.is_alive() and not t1.is_alive(), "a cancelled request wedged the scheduler"
    assert out1 == ref1[0]                            # the other stream finished with its solo reply


def _generate_parallel(engine, prompt, count, sampling, spec):
    out: list[int] = []
    engine.request.policy = spec
    engine.request.stop_eos = False
    engine.generate(list(prompt), count, sampling, lambda new: out.extend(new) or False, draft=True)
    return out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
