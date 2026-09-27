"""Profiles must round-trip a cue EXACTLY, and a missing one must not be an error.

The decisive check is the round trip: a profile that loads but predicts
differently would be worse than no profile at all, because the policy would
silently frame on a different cue than the one that was measured. So the test
compares predictions, not fields.
"""

import json

import numpy as np
import pytest

from steadycut.core import profiles
from steadycut.framing.ground_cue import GroundCue

rng = np.random.default_rng(0)
CUE = GroundCue(weights=rng.normal(size=6), mean=rng.normal(size=5),
                scale=rng.uniform(0.5, 2.0, size=5))


def frame(seed=1):
    return rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)


def test_round_trip_preserves_predictions(tmp_path):
    p = (profiles.Profile(pitch=-19.0, cue=CUE, notes={"ride": "x"})
         .record_source(__file__))
    p.fitted_on = {"frames": 7, "held_out_error": 0.082}
    profiles.save("abc", p, directory=tmp_path)
    back = profiles.load("abc", directory=tmp_path)

    assert back is not None
    assert back.pitch == -19.0
    assert back.notes == {"ride": "x"}
    assert back.fitted_on["frames"] == 7
    assert back.verified and back.matches(__file__)
    img = frame()
    assert np.allclose(back.cue.probability(img), CUE.probability(img))


def test_a_reexport_does_not_reuse_the_profile(tmp_path):
    """The gap this version closes: same path, different file."""
    src = tmp_path / "ride.insv"
    src.write_bytes(b"x" * 100)
    p = profiles.Profile(pitch=-19.0, cue=CUE).record_source(src)
    profiles.save(profiles.key_for(src), p, directory=tmp_path)

    assert profiles.load_for(src, directory=tmp_path).pitch == -19.0
    src.write_bytes(b"x" * 200)            # re-exported under the same name
    assert profiles.load_for(src, directory=tmp_path) is None


def test_quality_line_says_what_the_fit_was_worth():
    assert "pitch only" in profiles.Profile(pitch=-12.0).quality()
    fitted = profiles.Profile(pitch=-12.0, cue=CUE,
                              fitted_on={"frames": 12, "held_out_error": 0.082})
    assert "12 frames" in fitted.quality() and "0.082" in fitted.quality()


def test_quality_does_not_invent_a_cue():
    """Found by running a real profile: fitted_on present, cue absent, and the
    line claimed a cue fitted on zero frames. A judgeable line must not
    overstate what was measured."""
    p = profiles.Profile(pitch=-19.0, fitted_on={"frames": 0,
                                                  "held_out_error": None})
    assert p.cue is None
    assert "pitch only" in p.quality()


def test_the_store_holds_no_preferences(tmp_path):
    """Where the boundary should sit is a CHOICE, so it lives with the framing
    code and a per-activity override, not among a clip's measurements."""
    profiles.save("abc", profiles.Profile(pitch=-19.0), directory=tmp_path)
    d = json.loads((tmp_path / "abc.json").read_text())
    assert "target" not in d


def test_missing_profile_is_not_an_error(tmp_path):
    assert profiles.load("nothing-here", directory=tmp_path) is None


def test_corrupt_profile_reads_as_missing(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    assert profiles.load("bad", directory=tmp_path) is None


def test_save_leaves_no_partial_file(tmp_path):
    profiles.save("abc", profiles.Profile(pitch=1.0), directory=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["abc.json"]


def test_a_profile_without_a_cue_is_still_useful(tmp_path):
    """The pitch alone is worth storing: that is the whole of what the policy
    needs when the boundary is not measurable anywhere in the clip."""
    profiles.save("p", profiles.Profile(pitch=-12.5), directory=tmp_path)
    back = profiles.load("p", directory=tmp_path)
    assert back.pitch == -12.5 and back.cue is None


def test_key_is_stable_and_per_recording(tmp_path):
    a = profiles.key_for(tmp_path / "one.insv")
    assert a == profiles.key_for(tmp_path / "one.insv")
    assert a != profiles.key_for(tmp_path / "two.insv")
    assert len(a) == 16 and "/" not in a


def test_default_location_is_under_the_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert profiles.config_dir() == tmp_path / "steadycut"


def test_written_profile_is_plain_json(tmp_path):
    """Readable and diffable by a human: these are measurements, and someone
    should be able to see what was measured without the library."""
    profiles.save("abc", profiles.Profile(pitch=-19.0, cue=CUE), directory=tmp_path)
    d = json.loads((tmp_path / "abc.json").read_text())
    assert d["version"] == 2 and d["pitch"] == -19.0
    assert len(d["cue"]["weights"]) == 6
    assert "target" not in d       # a choice is not a measurement
