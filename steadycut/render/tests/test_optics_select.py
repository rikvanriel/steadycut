"""Pins for optics selection (Task 1.3): winner, enroll-new, refusal.

Footage extraction is monkeypatched out; what is pinned is the decision
logic (best-within-tolerance serves, else enroll/confirm) and the registry
writes. Real proof runs on footage under Task 1.4.
"""
import numpy as np
import pytest

from steadycut.ingest.cameras import registry as R
import steadycut.render.optics_select as S


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


@pytest.fixture
def fake_frames(monkeypatch):
    arr = np.full((64, 64, 3), 128, np.uint8)
    monkeypatch.setattr(S, "_frames", lambda source, times=None: [(arr, arr)])
    return arr


def _prof(fa, fb, **kw):
    d = {"fov_a": fa, "fov_b": fb, "fw": "v1"}
    d.update(kw)
    return d


def test_stored_winner_serves_without_fresh_fit(cfg, fake_frames,
                                                monkeypatch):
    R.record_measured("S1", model="Insta360 X4", fw="v1")
    R.record_optics("S1", _prof(183.5, 183.5, label="glass"), serial_known=True)
    # fresh grid fit lands on the stored profile: stored wins, not fresh.
    monkeypatch.setattr(S, "fit", lambda A, B: (183.5, 183.5, (1.0, 5.0), []))
    prof, fresh = S.select_optics("clip.insv", "S1", "v1", True)
    assert fresh is False
    assert prof["label"] == "glass"


def test_nothing_fits_enrolls_candidate(cfg, fake_frames, monkeypatch):
    R.record_measured("S2", model="Insta360 X4", fw="v1")
    R.record_optics("S2", _prof(180.0, 178.5, label="old"), serial_known=True)
    monkeypatch.setattr(S, "fit", lambda A, B: (185.0, 183.5, (0.5, 5.0), []))
    monkeypatch.setattr(S, "_score_profile",
                        lambda pairs, fa, fb: 2.0 if fa < 181 else 0.5)
    prof, fresh = S.select_optics("clip.insv", "S2", "v1", True)
    assert fresh is True
    assert prof["id"] == "S2-cand-2"
    assert prof["status"] == "candidate"


def test_served_candidate_confirms_on_second_recording(cfg, fake_frames,
                                                       monkeypatch):
    R.record_measured("S3", model="Insta360 X4", fw="v1")
    R.record_optics("S3", _prof(185.0, 183.5, label="new"), serial_known=True)
    monkeypatch.setattr(S, "fit", lambda A, B: (185.0, 183.5, (0.5, 5.0), []))
    monkeypatch.setattr(S, "_score_profile",
                        lambda pairs, fa, fb: 0.5)
    prof, fresh = S.select_optics("clip2.insv", "S3", "v1", True)
    assert fresh is False and prof["status"] == "confirmed"


def test_no_serial_serves_bench_without_registry(cfg, fake_frames):
    prof, fresh = S.select_optics("clip.insv", "missing", None, False)
    assert prof["id"] is None and fresh is True
    assert R.optics_for("missing") == []
