"""M0: register map invariants + the raw-index lint."""
import re
from pathlib import Path

import pytest

from bf_transformer.config import Config
from bf_transformer.registers import RegisterMap

REQUIRED = ["NODE_ID", "DIST", "EDGE_FROM", "EDGE_TO", "EDGE_W",
            "IS_NODE", "IS_EDGE", "CAND"]


def test_required_registers_exist():
    reg = RegisterMap()
    for name in REQUIRED:
        assert name in reg.names()
    assert any(n.startswith("SCRATCH_") for n in reg.names())


def test_slices_disjoint_and_cover_d_model():
    reg = RegisterMap()
    seen = set()
    for name in reg.names():
        s = reg.slice(name)
        dims = set(range(s.start, s.stop))
        assert not (dims & seen), f"register {name} overlaps another register"
        seen |= dims
    assert seen == set(range(reg.d_model))


def test_sizes_track_config():
    cfg = Config(n_max=5, d_max=3)
    reg = RegisterMap(cfg)
    assert reg.size("NODE_ID") == 5
    assert reg.size("EDGE_FROM") == 5
    assert reg.size("EDGE_TO") == 5
    assert reg.size("SLOT") == 3
    assert reg.size("SCRATCH_GATHER") == 3
    assert reg.d_model != RegisterMap(Config()).d_model


def test_scalar_idx_guard():
    reg = RegisterMap()
    with pytest.raises(ValueError):
        reg.idx("NODE_ID")
    with pytest.raises(IndexError):
        reg.idx_in("SLOT", 99)


def test_no_raw_indices_lint():
    """Construction/encode/model code must address the residual stream via
    the register map, never via integer literals on the d_model axis."""
    pkg = Path(__file__).resolve().parent.parent / "bf_transformer"
    # pattern: tensor indexing whose last axis is a bare integer literal,
    # e.g. stream[row, 17] or x[..., 5]
    raw = re.compile(r"\[[^\]\[]*,\s*\d+\s*\]")
    allowed = re.compile(r"(wq|wk|wv|wo)\[")  # head-internal dims are not registers
    for path in pkg.rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            m = raw.search(line)
            if m and not allowed.search(line):
                raise AssertionError(
                    f"raw residual-stream index in {path.name}:{lineno}: {line.strip()}")
