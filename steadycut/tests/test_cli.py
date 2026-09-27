"""CLI tests: parsing and refusal paths. No footage needed."""
from pathlib import Path

import pytest

from steadycut.steadycut import build_parser, main, refuse


def test_parser_requires_file_start_dur() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    a = build_parser().parse_args(["f.insv", "--start", "10", "--dur", "5"])
    assert (a.source, a.start, a.dur) == ("f.insv", 10.0, 5.0)
    assert a.tall is False and a.no_2d is False
    b = build_parser().parse_args(
        ["f.insv", "--start", "10", "--dur", "5", "--tall", "--pitch", "-14"])
    assert b.tall is True and b.pitch == -14.0


def test_refuse_returns_2_and_names_next_step(capsys) -> None:
    assert refuse("no dark mass", "re-try with --pitch") == 2
    out = capsys.readouterr().out
    assert "no dark mass" in out and "--pitch" in out


def test_non_camera_file_refuses(tmp_path: Path) -> None:
    fake = tmp_path / "plain.mp4"
    fake.write_bytes(b"\x00" * 512)
    rc = main([str(fake), "--start", "0", "--dur", "1",
               "--out", str(tmp_path / "o.mp4")])
    assert rc == 2
    assert not (tmp_path / "o.mp4").exists()  # no clip claimed


def test_the_corner_default_is_the_validated_one() -> None:
    """The tuning default is a measurement, so pin it in both of its homes.

    It was adopted because 0.15 Hz beat 0.5 Hz on every window of a
    multi-clip far-field validation; a change here should be a deliberate
    re-measurement, not a drift back to a rounder number. The library default
    and the CLI default are the same value, and must move together -- a tool
    that calls the library directly has to evaluate production.
    """
    import inspect
    from steadycut.core import pipeline as P
    a = build_parser().parse_args(["f.insv", "--start", "1", "--dur", "2"])
    lib = inspect.signature(P.build_gyro_path).parameters["corner_hz"].default
    assert a.corner == 0.15
    assert lib == a.corner


def _exdev_once(monkeypatch):
    """Make the first os.replace of the test fail as if crossing a device.

    The CLI stages its work in the system temp dir while the footage and the
    output normally live on a data disk, so this is the real case, not a
    contrived one: an EXDEV failure ("Invalid cross-device link") on the
    first move and a working one after.
    """
    import os
    real = os.replace
    state = {"n": 0}

    def fake(src, dst):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError(18, "Invalid cross-device link")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", fake)
    return state


def test_publish_survives_a_cross_device_move(tmp_path, monkeypatch) -> None:
    from steadycut.steadycut import publish

    staged = tmp_path / "stage" / "hybrid.mp4"
    staged.parent.mkdir()
    staged.write_bytes(b"stabilised bytes")
    out = tmp_path / "deliver" / "clip.mp4"
    out.parent.mkdir()

    _exdev_once(monkeypatch)
    publish(staged, out)
    assert out.read_bytes() == b"stabilised bytes"      # delivered anyway
    assert not staged.exists()                          # staged copy cleaned up
    assert not out.with_name(out.name + ".part").exists()   # no litter


def test_a_bare_replace_does_not_survive_it(tmp_path, monkeypatch) -> None:
    """Control: the call publish replaces has no fallback and does raise.

    Without this the test above only asserts "the bytes arrived", which a
    degenerate implementation satisfies by never moving at all.
    """
    import pytest

    staged = tmp_path / "stage" / "hybrid.mp4"
    staged.parent.mkdir()
    staged.write_bytes(b"stabilised bytes")
    out = tmp_path / "deliver" / "clip.mp4"
    out.parent.mkdir()

    _exdev_once(monkeypatch)
    with pytest.raises(OSError):
        staged.replace(out)
    assert not out.exists()
