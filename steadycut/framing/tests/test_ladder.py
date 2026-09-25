"""The ladder's own tests: it must be driven by whatever criterion it is given.

The sweep is the mount-independent half of the framing measurement, so the
tests here use a criterion that has nothing to do with helmets -- and one test
supplies two different criteria over the same renders to prove the sweep
follows the policy rather than a hardcoded notion of a subject.
"""
from pathlib import Path

import numpy as np
import pytest

from steadycut.framing import ladder
from steadycut.framing.policies import PitchSearch


def test_solve_pitch_interpolates_and_clamps() -> None:
    got = ladder.solve_pitch([-5.0, -12.0, -19.0], [1.0, 0.93, 0.83],
                             target=0.90)
    assert -19.0 < got < -12.0
    assert ladder.solve_pitch([-5.0, -12.0], [0.95, 0.93], target=0.90) == -12.0
    assert ladder.solve_pitch([-5.0, -12.0], [0.80, 0.70], target=0.90) == -5.0


def test_solve_pitch_with_nothing_measurable_is_nan() -> None:
    assert np.isnan(ladder.solve_pitch([-5.0, -12.0], [np.nan, np.nan], 0.90))


def test_brackets_tells_a_solved_ladder_from_a_clipped_one() -> None:
    """solve_pitch returns an END when the target is never met.

    A caller that only asks "did the criterion return a number?" cannot tell a
    solved pitch from one nobody checked -- which is how a -5.0 pitch got
    reported as usable on a window whose dark-mass edge never got past 0.78.
    """
    # Real numbers from a May 26 window: 0.5717 (pitch -26) .. 0.7778 (-5).
    assert ladder.brackets({-26.0: 0.5717, -12.0: 0.7003, -5.0: 0.7778},
                           0.90) is False
    assert ladder.brackets({-26.0: 0.5717, -12.0: 0.91, -5.0: 0.7778},
                           0.90) is True
    assert ladder.brackets({-5.0: 0.9999}, 0.90) is False
    assert ladder.brackets({-5.0: float("nan")}, 0.90) is False
    assert ladder.brackets({}, 0.90) is False


def test_the_workdir_is_created_when_missing(tmp_path) -> None:
    """The caller passes a fresh temp subdirectory, so it does NOT exist yet.

    ffmpeg fails with a bare exit status when the output directory is absent,
    which made the whole automatic framing path refuse the first time it was
    exercised without an explicit --pitch.
    """
    target = tmp_path / "framing" / "nested"
    assert not target.exists()
    out = ladder._workdir(target)
    assert out == target and out.is_dir()


def _fake_sweep(monkeypatch, value_of_pitch):
    """Render nothing; make the sweep 'measure' a value from the pitch.

    The ladder's renders are stubbed out, and measure_frames reads the pitch
    back out of the render's filename, so a test can assert which pitches were
    tried and what the criterion did with them.
    """
    from steadycut.core import pipeline as P

    monkeypatch.setattr(P, "render_constant", lambda spec, dst: dst)

    def fake_frames(video, w=ladder.MEASURE_W, h=ladder.MEASURE_H):
        pitch = float(Path(video).stem[1:])      # "p5.mp4" -> 5 (ladder is negative)
        return np.full((1, 1, 1), value_of_pitch(-pitch))

    monkeypatch.setattr(ladder, "measure_frames", fake_frames)


def test_the_sweep_follows_the_criterion_it_is_given(monkeypatch, tmp_path) -> None:
    """Same renders, two criteria, two answers: the sweep is policy-driven.

    A sweep with a helmet hardcoded into it would give the same pitch for both.
    Each criterion reads the value the fake render encodes for the pitch it was
    asked to try, so the ladder is doing nothing but driving a measurement and
    landing it on a target.
    """
    _fake_sweep(monkeypatch, lambda pitch: (pitch + 30.0) / 100.0)
    rising = PitchSearch(criterion=lambda frames: float(frames[0, 0, 0]),
                         target=0.20, label="rising", ladder=(-5.0, -12.0))
    rising_pitch, rising_info = ladder.solve("x.insv", 0.0, rising,
                                             workdir=tmp_path)

    _fake_sweep(monkeypatch, lambda pitch: 1.0 - (pitch + 30.0) / 100.0)
    falling = PitchSearch(criterion=lambda frames: float(frames[0, 0, 0]),
                          target=0.20, label="falling", ladder=(-5.0, -12.0))
    falling_pitch, falling_info = ladder.solve("x.insv", 0.0, falling,
                                               workdir=tmp_path)

    # rising: -5 measures 0.25 and -12 measures 0.18, so the 0.20 target is
    # inside the ladder and the answer is interpolated between the two.
    assert rising_info["edges"][-5.0] > rising_info["edges"][-12.0]
    assert rising_info["usable"] is True
    assert -12.0 < rising_pitch < -5.0
    # falling: the same target now sits outside the ladder, so the answer is a
    # clamped END -- unusable, and reported as such.
    assert falling_info["edges"][-5.0] < falling_info["edges"][-12.0]
    assert falling_info["usable"] is False
    assert falling_pitch == -5.0
    assert rising_pitch != falling_pitch
    assert rising_info["criterion"] == "rising"
    assert falling_info["criterion"] == "falling"


def test_a_clipped_ladder_is_refused_with_its_range(monkeypatch, tmp_path) -> None:
    """The refusal has to carry the numbers, not just "failed"."""
    _fake_sweep(monkeypatch, lambda pitch: 0.10)      # never near the target
    search = PitchSearch(criterion=lambda frames: float(frames[0, 0, 0]),
                         target=0.90, label="flat", ladder=(-5.0, -12.0))
    _, info = ladder.solve("x.insv", 0.0, search, workdir=tmp_path)
    assert info["usable"] is False
    assert info["bracketed"] is False
    assert "0.90" in info["reason"] and "flat" in info["reason"]


def test_an_unmeasurable_pitch_is_refused(monkeypatch, tmp_path) -> None:
    _fake_sweep(monkeypatch, lambda pitch: np.nan)
    search = PitchSearch(criterion=lambda frames: float(frames[0, 0, 0]),
                         target=0.90, label="nothing", ladder=(-5.0, -12.0))
    pitch, info = ladder.solve("x.insv", 0.0, search, workdir=tmp_path)
    assert info["usable"] is False
    assert np.isnan(pitch)
    assert "could not be measured" in info["reason"]
