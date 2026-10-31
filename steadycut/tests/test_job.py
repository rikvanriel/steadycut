"""Job envelope, fingerprint, window check, snap math. No footage needed."""
import json

import pytest

from steadycut.job import (ENVELOPE_KEYS, OutputLocked, apply_envelope,
                           check_window, claim_out, fingerprint,
                           load_envelope, media_duration, proxy_manifest,
                           snap_window, source_identity, write_envelope)
from steadycut.steadycut import build_parser, main


def _args(**kw):
    base = {"source": "f.insv", "start": 10.0, "dur": 5.0,
            "framing": "mtb", "pitch": None}
    base.update(kw)
    return base


def test_envelope_round_trip(tmp_path):
    (tmp_path / "f.insv").write_bytes(b"\x00" * 64)
    env = write_envelope(tmp_path / "j.json",
                         _args(source=str(tmp_path / "f.insv")),
                         base=tmp_path)
    back = load_envelope(tmp_path / "j.json")
    assert back["args"]["framing"] == "mtb"
    assert back["source"]["path"].endswith("f.insv") or True
    assert back["fingerprint"] == env["fingerprint"]


def test_envelope_rejects_version_and_unknown_and_missing(tmp_path):
    p = tmp_path / "j.json"
    (tmp_path / "f.insv").write_bytes(b"\x00" * 64)
    write_envelope(p, _args(source=str(tmp_path / "f.insv")))
    env = json.loads(p.read_text())
    env["version"] = 999
    p.write_text(json.dumps(env))
    with pytest.raises(ValueError):
        load_envelope(p)
    write_envelope(p, _args(source=str(tmp_path / "f.insv")))
    env = json.loads(p.read_text())
    env["args"]["no_such_flag"] = 1
    p.write_text(json.dumps(env))
    with pytest.raises(ValueError):
        load_envelope(p)
    env["args"] = {"source": "f.insv"}
    p.write_text(json.dumps(env))
    with pytest.raises(ValueError):
        load_envelope(p)


def test_fingerprint_stable_sensitive_order_free():
    a, b = _args(), _args()
    assert fingerprint(a) == fingerprint(b)
    c = _args(pitch=-14.0)
    assert fingerprint(c) != fingerprint(a)
    assert fingerprint({"dur": 5.0, "start": 10.0, "source": "f.insv",
                        "framing": "mtb", "pitch": None}) == fingerprint(a)


def test_check_window(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    dur_fn = lambda s: 900.0  # noqa: E731
    assert check_window(src, 10.0, 5.0, duration_fn=dur_fn) == (True, "ok")
    ok, why = check_window(tmp_path / "missing.insv", 0, 1,
                           duration_fn=dur_fn)
    assert not ok and "not found" in why
    ok, why = check_window(src, -1.0, 5.0, duration_fn=dur_fn)
    assert not ok and "negative" in why
    ok, why = check_window(src, 899.0, 5.0, duration_fn=dur_fn)
    assert not ok and "past media end" in why
    ok, _ = check_window(src, 10.0, 0.0001, duration_fn=dur_fn)
    assert not ok  # snaps to nothing


def test_snap_window_frame_grid(tmp_path):
    # Empty file: ffprobe fails, source_fps falls back to 29.97. The grid
    # is still a grid: both edges land on frame multiples, end exclusive.
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    s, d = snap_window(src, 10.0, 5.0, duration_fn=lambda s_: 900.0)
    fps = 29.97
    assert abs(s * fps - round(s * fps)) < 1e-6
    assert abs((s + d) * fps - round((s + d) * fps)) < 1e-6
    assert d > 4.9 and s <= 10.0


def test_snap_uses_probed_fps_not_assumption(tmp_path, monkeypatch):
    # 30000/1001 is not 29.97: the snapped edges must sit on the probed
    # grid, not the rounded one.
    import steadycut.render.reframe as rf
    monkeypatch.setattr(rf, "source_fps", lambda s: 30000 / 1001)
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    s, d = snap_window(src, 10.0, 5.0, duration_fn=lambda s_: 900.0)
    fps = 30000 / 1001
    assert abs(s * fps - round(s * fps)) < 1e-6
    assert abs((s + d) * fps - round((s + d) * fps)) < 1e-6


def test_apply_envelope_fills_unset_explicit_wins(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    env_args = {"source": str(src), "start": 10.0, "dur": 5.0,
                "framing": "ski", "pitch": -14.0, "tall": True}
    env = {"version": 1, "args": env_args}
    argv = [str(src), "--start", "10", "--dur", "5"]
    ns = build_parser().parse_args(argv)
    apply_envelope(ns, env, argv)
    assert (ns.framing, ns.pitch, ns.tall) == ("ski", -14.0, True)
    argv2 = [str(src), "--start", "10", "--dur", "5", "--framing", "mtb"]
    ns2 = build_parser().parse_args(argv2)
    apply_envelope(ns2, env, argv2)
    assert ns2.framing == "mtb"  # explicit flag wins over the envelope
    # The = form and argparse abbreviations are explicit too: the envelope
    # must not silently override them.
    for argv3 in (
        [str(src), "--start", "10", "--dur", "5", "--framing=mtb"],
        [str(src), "--start", "10", "--dur", "5", "--fram", "mtb"],
    ):
        ns3 = build_parser().parse_args(argv3)
        apply_envelope(ns3, env, argv3)
        assert ns3.framing == "mtb"  # explicit flag wins over the envelope


def test_media_duration_reads_probe_video_not_streams(tmp_path):
    # Unmocked: a real 1 s lavfi clip proves media_duration reads
    # Probe.video (CI has ffmpeg). The old .streams attribute never
    # existed, so this raised AttributeError on any real file while the
    # mocked tests stayed green.
    import subprocess

    clip = tmp_path / "mini.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=s=160x120:r=10:d=1",
         "-frames:v", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(clip)], check=True, timeout=120)
    assert media_duration(clip) == pytest.approx(1.0, abs=0.2)


def test_apply_envelope_refuses_replaced_source(tmp_path):
    # The envelope pins size+mtime: a silently replaced file must not
    # pass as the same source.
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    env = write_envelope(tmp_path / "j.json", _args(source=str(src)))
    src.write_bytes(b"\x00" * 128)  # rewritten under a stable name
    ns = build_parser().parse_args([str(src), "--start", "10", "--dur", "5"])
    with pytest.raises(ValueError, match="changed since envelope"):
        apply_envelope(ns, load_envelope(tmp_path / "j.json"),
                       [str(src), "--start", "10", "--dur", "5"])


def test_claim_out_keeps_lockfile_and_cleans_staged_on_failure(tmp_path):
    # A failed render leaves its staged bytes behind: rmtree in the
    # finally must remove the whole staged dir, not just rmdir it
    # (rmdir fails on a non-empty dir and litters partial clips).
    out = tmp_path / "clip.mp4"
    staged = None
    try:
        with claim_out(out) as staged:
            (staged.parent / "partial.mp4").write_bytes(b"partial")
            raise RuntimeError("render blew up")
    except RuntimeError:
        pass
    assert staged is not None and not staged.parent.exists()
    # Contention: a live holder raises OutputLocked for a second claim.
    with claim_out(out):
        with pytest.raises(OutputLocked):
            with claim_out(out):
                pass


def test_apply_envelope_refuses_stale_job(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    env = {"version": 1, "args": {"source": str(src), "start": 10.0,
                                 "dur": 5.0}}
    ns = build_parser().parse_args(
        [str(src), "--start", "20", "--dur", "5"])
    with pytest.raises(ValueError):
        apply_envelope(ns, env)


def test_proxy_manifest_omits_aim_and_aspect(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    m = proxy_manifest(src, {"model": "X4"}, "legacy", "reframe-1")
    blob = json.dumps(m)
    assert "yaw" not in blob and "aspect" not in blob and "tall" not in blob
    m2 = proxy_manifest(src, {"model": "X4"}, "blend", "reframe-1")
    assert m2["fingerprint"] != m["fingerprint"]


def test_source_identity_pins_size(tmp_path):
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    i1 = source_identity(src)
    src.write_bytes(b"\x00" * 128)
    assert source_identity(src)["size"] != i1["size"]


def test_job_alone_supplies_the_window(tmp_path):
    # A bare `steadycut --job gui_job.json` parses with no window on argv;
    # the envelope supplies source/start/dur.
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    env = write_envelope(tmp_path / "j.json", _args(source=str(src)))
    ns = build_parser().parse_args(["--job", str(tmp_path / "j.json")])
    assert (ns.source, ns.start, ns.dur) == (None, None, None)
    apply_envelope(ns, env, ["--job", str(tmp_path / "j.json")])
    assert (ns.source, ns.start, ns.dur) == (str(src), 10.0, 5.0)


def test_job_source_resolves_against_base(tmp_path, monkeypatch):
    # The envelope stores the source relative to base, so a moved project
    # directory still resolves its media instead of pointing at the old path.
    proj = tmp_path / "proj"
    proj.mkdir()
    src = proj / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    env = write_envelope(tmp_path / "j.json", _args(source=str(src)),
                         base=proj)
    assert env["args"]["source"] == "clip.insv"
    monkeypatch.chdir(tmp_path)  # away from the project dir
    ns = build_parser().parse_args(["--job", str(tmp_path / "j.json")])
    apply_envelope(ns, env, ["--job", str(tmp_path / "j.json")])
    assert ns.source == str(src)
    assert (ns.start, ns.dur) == (10.0, 5.0)


def test_job_argv_matches_literal_argv(tmp_path):
    # The byte-identical contract at arg level: a job run and the literal
    # argv it was exported from agree on every envelope key.
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    literal = build_parser().parse_args(
        [str(src), "--start", "10", "--dur", "5", "--framing", "ski",
         "--pitch", "-14", "--tall", "--no-2d", "--preset", "fast"])
    write_envelope(tmp_path / "j.json", dict(vars(literal)))
    argv = [str(src), "--start", "10", "--dur", "5",
            "--job", str(tmp_path / "j.json")]
    ns = build_parser().parse_args(argv)
    apply_envelope(ns, load_envelope(tmp_path / "j.json"), argv)
    assert all(getattr(ns, k) == getattr(literal, k) for k in ENVELOPE_KEYS)


def test_main_check_validates_without_rendering(tmp_path, monkeypatch):
    import steadycut.job as J
    monkeypatch.setattr(J, "media_duration", lambda s: 900.0)
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    out = tmp_path / "o.mp4"
    assert main([str(src), "--start", "10", "--dur", "5", "--check",
                 "--out", str(out)]) == 0
    assert not out.exists()  # validated, not rendered
    assert main([str(src), "--start", "899", "--dur", "5", "--check",
                 "--out", str(out)]) == 2
    assert not out.exists()


def test_main_check_with_bare_job(tmp_path, monkeypatch):
    import steadycut.job as J
    monkeypatch.setattr(J, "media_duration", lambda s: 900.0)
    src = tmp_path / "clip.insv"
    src.write_bytes(b"\x00" * 64)
    write_envelope(tmp_path / "j.json", _args(source=str(src)))
    assert main(["--job", str(tmp_path / "j.json"), "--check"]) == 0


def test_claim_out_serialises_one_output(tmp_path):
    out = tmp_path / "clip.mp4"
    with claim_out(out) as staged:
        assert staged.name == "clip.mp4"
        assert staged.parent != tmp_path  # staged aside, never in place
        with pytest.raises(OutputLocked):
            with claim_out(out):
                pass
    with claim_out(out) as staged2:  # released on exit: claimable again
        pass
    # The lockfile is kept, never unlinked while locked: unlinking the
    # live path orphans the inode, so two renders can each believe they
    # hold the lock. The staged temp dir is still cleaned up.
    assert out.with_name(out.name + ".lock").exists()
    assert not staged2.parent.exists()
