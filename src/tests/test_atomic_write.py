"""core.atomic_write.write_parquet_atomic: the one way the gate-input caches (Yahoo, FRED, VX continuous) are written.
Run: cd src && python3 -m pytest tests/test_atomic_write.py -q"""
import fnmatch
import os
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import atomic_write as aw

REPO = Path(__file__).resolve().parents[2]


def _frame(value):
    return pd.DataFrame({"Close": [value]}, index=pd.DatetimeIndex(["2026-11-04"]))


def _spy_replace(monkeypatch):
    real_replace, replaced = os.replace, []

    def spy(src, dst):
        replaced.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(aw.os, "replace", spy)
    return replaced


# Writer B (a manual gate-refresh) runs its whole write between writer A's (the launchd end-session) temp write and its
# os.replace. A shared temp name would let B move A's temp away, so A's os.replace raises.
def test_two_interleaved_writers_both_complete(tmp_path, monkeypatch):
    path = tmp_path / "yahoo_GSPC.parquet"
    a, b = _frame(1.0), _frame(2.0)
    write = pd.DataFrame.to_parquet

    def to_parquet(self, target, *args, **kwargs):
        write(self, target, *args, **kwargs)
        if self is a:
            aw.write_parquet_atomic(b, path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    replaced = _spy_replace(monkeypatch)
    aw.write_parquet_atomic(a, path)
    assert pd.read_parquet(path).loc["2026-11-04", "Close"] == 1.0
    assert os.listdir(tmp_path) == [path.name]
    assert len({src for src, _ in replaced}) == 2
    assert all(src.parent == dst.parent == tmp_path for src, dst in replaced)


# A temp name already on disk belongs to another writer: it is skipped (O_EXCL), never truncated or moved.
def test_a_taken_temp_name_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "yahoo_GSPC.parquet"
    tokens = iter(["aaaa", "bbbb"])
    monkeypatch.setattr(aw.secrets, "token_hex", lambda n: next(tokens))
    other = tmp_path / ".yahoo_GSPC.parquet.aaaa.tmp"
    other.write_bytes(b"another writer")
    aw.write_parquet_atomic(_frame(2.0), path)
    assert other.read_bytes() == b"another writer"
    assert pd.read_parquet(path).loc["2026-11-04", "Close"] == 2.0
    assert sorted(os.listdir(tmp_path)) == sorted([other.name, path.name])


def test_failed_write_keeps_the_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    path = tmp_path / "yahoo_GSPC.parquet"
    _frame(1.0).to_parquet(path)
    before = path.read_bytes()

    def half_write(self, target, *args, **kwargs):
        Path(target).write_bytes(b"PAR1 half")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", half_write)
    with pytest.raises(OSError, match="disk full"):
        aw.write_parquet_atomic(_frame(2.0), path)
    assert path.read_bytes() == before
    assert os.listdir(tmp_path) == [path.name]


# Ctrl-C during a manual gate-refresh is a KeyboardInterrupt, not an Exception: the temp must still go.
def test_interrupted_write_leaves_no_temp(tmp_path, monkeypatch):
    path = tmp_path / "yahoo_GSPC.parquet"

    def interrupted(self, target, *args, **kwargs):
        Path(target).write_bytes(b"PAR1 half")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    with pytest.raises(KeyboardInterrupt):
        aw.write_parquet_atomic(_frame(2.0), path)
    assert os.listdir(tmp_path) == []


def test_replacing_keeps_the_old_files_mode(tmp_path):
    path = tmp_path / "yahoo_GSPC.parquet"
    _frame(1.0).to_parquet(path)
    os.chmod(path, 0o640)
    aw.write_parquet_atomic(_frame(2.0), path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    assert pd.read_parquet(path).loc["2026-11-04", "Close"] == 2.0


def test_a_new_file_gets_the_umask_mode(tmp_path):
    path = tmp_path / "yahoo_GSPC.parquet"
    umask = os.umask(0o027)
    try:
        aw.write_parquet_atomic(_frame(2.0), path)
    finally:
        os.umask(umask)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o666 & ~0o027


def test_no_file_descriptor_leaks(tmp_path):
    path = tmp_path / "yahoo_GSPC.parquet"
    aw.write_parquet_atomic(_frame(1.0), path)
    before = len(os.listdir("/dev/fd"))
    for value in range(5):
        aw.write_parquet_atomic(_frame(float(value)), path)
    assert len(os.listdir("/dev/fd")) == before


# A temp left by a killed process must never be staged by a data commit.
def test_temp_name_is_gitignored(tmp_path, monkeypatch):
    path = tmp_path / "VX1.parquet"
    replaced = _spy_replace(monkeypatch)
    aw.write_parquet_atomic(_frame(1.0), path)
    patterns = [ln.strip() for ln in (REPO / ".gitignore").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    temp = replaced[0][0].name
    assert temp != path.name
    assert any(fnmatch.fnmatchcase(temp, p) for p in patterns), (temp, patterns)
    assert not any(fnmatch.fnmatchcase(path.name, p) for p in patterns)


# VX1 is a gate input (C1/C6); the builder replaces its parquet instead of rewriting it in place.
def test_vx_continuous_write_replaces_never_truncates(tmp_path, monkeypatch):
    from core import vx_continuous_contract_builder as vxm
    vx_dir, cache_dir = tmp_path / "CBOE_VX_ALL", tmp_path / "vx_continuous"
    vx_dir.mkdir()
    builder = vxm.VXContinuousContractBuilder(cboe_vx_dir=str(vx_dir), cache_dir=str(cache_dir))
    path = cache_dir / "VX1.parquet"
    pd.DataFrame({"settle": [20.0]}, index=pd.DatetimeIndex(["2026-11-04"])).to_parquet(path)
    rebuilt = pd.DataFrame({"settle": [20.0, 21.5]}, index=pd.DatetimeIndex(["2026-11-04", "2026-11-05"]))
    monkeypatch.setattr(builder, "_build_continuous_incremental", lambda pos, start: rebuilt)
    before = path.read_bytes()
    with open(path, "rb") as reader:
        out = builder.get_continuous_contract(1, force_rebuild=True)
        assert out.index[-1] == pd.Timestamp("2026-11-05")
        assert pd.read_parquet(path).loc["2026-11-05", "settle"] == 21.5
        assert reader.read() == before
    assert os.listdir(cache_dir) == [path.name]
