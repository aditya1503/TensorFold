"""The GLM tree's integrity, written after the first DSpark deploy crashed on it: a build from the fork's bare
branch lacked the recipe's patches, and ``app.py`` called ``KeptReasoning`` without importing it — a runtime
``NameError`` that no draft test and no import smoke caught. These tests pin the classes of that failure:

1. ``ruff --select F821`` over the family: no name is used that the tree does not define (the app.py bug).
2. Every module of ``glm5_next``'s CUDA side imports (fake triton, real torch): no missing file or import.
3. The shipped RedHatAI DSpark checkpoint's config.json parses into what the loader reads, and the two ranks'
   parity input (``drafter_taps``) counts its six target layers.

The GPU-only checks stay in tests/cuda/ (the DGX Sparks): the block pass's bits, verify acceptance, --parallel.
"""

import json
from pathlib import Path

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)

pytestmark = pytest.mark.torch

FAMILY = Path(__file__).resolve().parents[1] / "src" / "tensorfold" / "families" / "glm5_next"
DSPARK_HUB = Path.home() / ".cache" / "huggingface" / "hub" / "models--RedHatAI--GLM-5.3-Flash-speculator.dspark-preview"


def family_modules() -> list[str]:
    """glm5_next's CUDA-side modules and its package root (the Mac side needs MLX, not imported here)."""

    names = ["tensorfold.families.glm5_next"]
    for path in sorted((FAMILY / "cuda").glob("*.py")):
        if path.stem != "__init__":
            names.append(f"tensorfold.families.glm5_next.cuda.{path.stem}")
    return names


def test_the_family_defines_every_name_it_uses():  # the app.py class: a bare NameError at startup
    """ruff's F821 over glm5_next: no undefined names — the check that would have caught KeptReasoning."""

    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "ruff", "check", "--select", "F821", "--output-format", "concise",
                         str(FAMILY)], capture_output=True, text=True)
    if out.returncode == 1 and "No module named" in out.stderr:
        pytest.skip("ruff is not installed")
    assert out.returncode == 0, out.stdout + out.stderr


def test_every_cuda_module_imports(allocations):  # noqa: F811
    """glm5_next's CUDA modules all import with fake triton (the build's own smoke imports only three)."""

    import importlib

    for name in family_modules():
        importlib.import_module(name)


def test_the_shipped_dspark_checkpoint_parses(allocations):  # noqa: F811
    """The RedHatAI preview's config.json, as the loader reads it (skipped where the checkpoint is not pulled)."""

    snapshots = DSPARK_HUB / "snapshots"
    if not snapshots.is_dir():
        pytest.skip("RedHatAI/GLM-5.3-Flash-speculator.dspark-preview is not in the local cache")
    import importlib

    mod = importlib.import_module("tensorfold.families.glm5_next.cuda.dspark")
    rev = next((snapshots / d).name for d in sorted(snapshots.iterdir()) if (snapshots / d / "config.json").is_file())
    folder = snapshots / rev
    cfg = json.loads((folder / "config.json").read_text())
    assert mod.drafter_kind(folder) == "dspark"
    c = mod.read_speculators_config(cfg)
    assert (c["block"], c["mask_id"], c["vocab"]) == (8, 154856, 154880)
    assert c["taps"] == (19, 27, 31, 35, 39, 43)     # the config's aux ids [20, 28, 32, 36, 40, 44] in this engine's
    # post-layer tap convention (HF output_hidden_states id L = the state entering layer L = this engine's L - 1)
    assert (c["markov_rank"], c["confidence"], c["anchor_drafts"], c["causal"]) == (256, True, True, True)
    assert c["hidden"] + c["markov_rank"] == 4352                # the confidence head's input width
    assert mod.drafter_taps(folder) == 6                          # the two-rank parity input
    import torch
    from safetensors import safe_open

    with safe_open(str(folder / "model.safetensors"), framework="pt", device="cpu") as f:
        names = set(f.keys())
    assert "markov_head.markov_w1.weight" in names and "confidence_head.proj.weight" in names
    assert len(names) == 64                                       # the loader's every tensor, none missing
