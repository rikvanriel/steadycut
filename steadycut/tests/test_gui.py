"""GUI translation tests: envelope, argv, refusals, proxy cache, runner.

Qt-free: none of these import PySide6, so they run in the base install.
"""

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import steadycut
from steadycut.gui import backend as B
from steadycut.gui import runner as R
from steadycut.job import ENVELOPE_KEYS, load_envelope


def _state(**kw):
    kw.setdefault("source", "clip.insv")
    kw.setdefault("start", 10.0)
    kw.setdefault("dur", 5.0)
    kw.setdefault("out", "/tmp/clip_10s_wide.mp4")
    return B.GuiState(**kw)


def test_render_args_use_envelope_vocabulary_only():
    args = B.state_to_render_args(_state(framing="ski", pitch=-14.0,
                                         tall=True))
    assert set(args) <= set(ENVELOPE_KEYS)
    for need in ("source", "start", "dur"):
        assert need in args


def test_envelope_round_trip_through_panel_state(tmp_path):
    (tmp_path / "clip.insv").write_bytes(b"\x00" * 64)
    state = _state(source=str(tmp_path / "clip.insv"), framing="ski",
                   pitch=-14.0, yaw_bias=2.5)
    env = B.write_job(state, tmp_path / "gui_job.json", base=tmp_path)
    assert env["version"] == 1
    back = load_envelope(tmp_path / "gui_job.json")
    assert back["args"]["framing"] == "ski"
    assert back["args"]["yaw_bias"] == 2.5
    assert back["fingerprint"] == env["fingerprint"]
    # Stored relative to the project root, like the backend contract.
    assert back["args"]["source"] == "clip.insv"


def test_load_job_fills_the_panel(tmp_path):
    state = _state()
    back = B.fill_state_from_args(
        state, {"framing": "ski", "pitch": -14.0, "no_such_flag": 1})
    assert (back.framing, back.pitch) == ("ski", -14.0)


def test_literal_argv_matches_job_argv_namespace(tmp_path):
    """Copy CLI renders the same file as the Render button's --job run."""
    from steadycut.job import apply_envelope, load_envelope
    from steadycut.steadycut import build_parser

    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    state = _state(source=str(src), framing="ski", pitch=-14.0, tall=True,
                   yaw_bias=2.5, no_2d=True, preset="fast")
    env_path = tmp_path / "j.json"
    B.write_job(state, env_path, base=tmp_path)
    literal = B.literal_argv(state)
    rest = literal[1:] if literal[0] == "steadycut" else literal[3:]
    ns_literal = build_parser().parse_args(rest)
    job = B.job_argv(state, env_path)
    jrest = job[3:]
    ns_job = build_parser().parse_args(jrest)
    apply_envelope(ns_job, load_envelope(env_path), jrest)
    assert all(getattr(ns_job, k) == getattr(ns_literal, k)
               for k in ENVELOPE_KEYS)


def test_literal_argv_omits_defaults_but_keeps_window(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    argv = B.literal_argv(_state(source=str(src)))
    text = " ".join(argv)
    assert "--framing" not in text and "--pitch" not in text
    assert "--tall" not in text and "--no-2d" not in text
    assert "--start" in text and "--dur" in text
    assert str(src) in text


def test_job_argv_carries_window_job_and_out(tmp_path):
    state = _state()
    argv = B.job_argv(state, tmp_path / "gui_job.json")
    assert "--job" in argv and str(tmp_path / "gui_job.json") in argv
    assert argv[3:6] == [state.source, "--start", repr(10.0)]
    assert "--out" in argv and state.out in argv


def test_extract_refusals_pairs_msg_with_hint():
    log = ("CLI| camera X via trailer\n"
           "REFUSE| no framing policy named nope\n"
           "REFUSE| known policies: mtb, ski\n"
           "REFUSE| window end past media end\n"
           "REFUSE| pick a window inside the file\n")
    assert B.extract_refusals(log) == [
        ("no framing policy named nope", "known policies: mtb, ski"),
        ("window end past media end", "pick a window inside the file"),
    ]
    assert B.extract_refusals("CLI| wrote o.mp4\n") == []


def test_stage_tracking_follows_the_pipeline():
    assert B.stage_of("CLI| camera X4 via trailer") == "identify"
    assert B.stage_of("CLI| framing pitch -12.0 (policy mtb)") == "framing"
    assert B.stage_of("CLI| time sync 3.1s") == "sync"
    assert B.stage_of("CLI| wrote /tmp/o.mp4") == "done"
    assert B.stage_of("garbage line") is None
    assert B.stage_fraction("identify") is None
    frac = B.stage_fraction("sync")
    assert frac is not None and 0.0 < frac < 1.0
    assert B.stage_fraction("done") == 1.0


def test_proxy_cache_goes_stale_when_the_file_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    manifest = B.current_proxy_manifest(str(src), {"model": "X4"},
                                        "legacy")
    video, meta = B.proxy_paths(manifest["fingerprint"])
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(manifest))
    assert B.proxy_is_fresh(meta, str(src), {"model": "X4"}, "legacy")
    assert not B.proxy_is_fresh(meta, str(src), {"model": "X4"}, "blend")
    src.write_bytes(b"\x00" * 128)  # re-exported under a stable name
    assert not B.proxy_is_fresh(meta, str(src), {"model": "X4"}, "legacy")
    assert not B.proxy_is_fresh(tmp_path / "missing.json", str(src),
                                {"model": "X4"}, "legacy")


def test_time_fields_round_trip():
    assert B.fmt_time(372.5) == "06:12.5"
    assert B.parse_time("06:12.5") == pytest.approx(372.5)
    assert B.parse_time("372.5") == pytest.approx(372.5)
    assert B.parse_time("6:12") == pytest.approx(372.0)


def test_proxy_span_is_the_first_minute():
    assert B.proxy_span(900.0) == (0.0, 60.0)
    assert B.proxy_span(20.0) == (0.0, 20.0)


def test_default_out_falls_back_from_read_only_media(tmp_path):
    import stat

    locked = tmp_path / "locked"
    locked.mkdir()
    src = locked / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        out = B.default_out(str(src), 10.0)
        assert not out.startswith(str(locked))
        assert out.endswith("clip_10s_wide.mp4")
    finally:
        locked.chmod(0o755)


def test_check_state_preflights_without_rendering(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    dur_fn = lambda s: 900.0  # noqa: E731
    ok, _ = B.check_state(_state(source=str(src)), duration_fn=dur_fn)
    assert ok
    ok, why = B.check_state(_state(source=str(tmp_path / "gone.insv")),
                            duration_fn=dur_fn)
    assert not ok and "not found" in why
    ok, why = B.check_state(_state(source=str(src), start=899.0),
                            duration_fn=dur_fn)
    assert not ok and "past media end" in why


def test_no_qt_outside_the_gui_package():
    """The base install stays Qt-free: grep, not import state."""
    ROOT = Path(steadycut.__file__).parent
    bad = []
    for path in sorted(ROOT.rglob("*.py")):
        if "gui" in path.parts or "tests" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] in ("PySide6", "PyQt6", "PyQt5")
                   for n in names):
                bad.append(str(path))
    assert not bad, bad


def test_gui_backend_imports_without_qt(tmp_path):
    """backend/runner serve the base install: importable with Qt blocked.

    A subprocess with a PySide6-blocking import hook proves the CLI side
    never touches Qt, independent of what this session already imported.
    """
    root = Path(steadycut.__file__).parent.parent
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class _BlockQt:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'PySide6' else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('Qt blocked for hygiene check')\n"
        "sys.meta_path.insert(0, _BlockQt())\n")
    code = ("import steadycut.gui.backend, steadycut.gui.runner, "
            "steadycut.job, steadycut.steadycut; print('qt-free ok')")
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(root))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "qt-free ok" in proc.stdout


@pytest.mark.skipif(os.name != "posix", reason="process groups are posix")
def test_cancel_kills_the_whole_group_including_ffmpeg(tmp_path):
    """A render's ffmpeg child shares the job's group and must die too."""
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, time\n"
        "subprocess.Popen(['sleep', '60'])\n"
        "time.sleep(60)\n")
    log = tmp_path / "job.log"
    proc, fh = R.start_job([sys.executable, str(parent)], log)
    try:
        pgid = os.getpgid(proc.pid)
        time.sleep(0.5)
        assert R.group_pids(proc.pid), "child never joined the group"
        outcome = R.cancel(proc, grace_s=2.0)
        assert outcome in ("term", "kill")
        assert proc.poll() is not None
        time.sleep(0.3)
        assert not R.group_pids(pgid), "group survived cancel"
    finally:
        fh.close()
        if proc.poll() is None:
            proc.kill()


@pytest.mark.skipif(os.name != "posix", reason="process groups are posix")
def test_cancel_on_an_exited_job_is_done(tmp_path):
    log = tmp_path / "job.log"
    proc, fh = R.start_job(["true"], log)
    try:
        assert proc.wait(timeout=10) == 0
        assert R.cancel(proc) == "done"
    finally:
        fh.close()
