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
