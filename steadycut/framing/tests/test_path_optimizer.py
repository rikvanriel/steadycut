"""The path optimiser's invariants, tested as properties rather than examples.

Three properties matter, and the third is the one the whole formulation rests on:

1. UNCONSTRAINED IS IDENTITY. With no active constraint the answer must be the
   existing path exactly. If it is not, the optimiser is not minimising movement
   and every measurement it produces is untrustworthy.
2. THE SKY BOUND IS ONE-SIDED. It may only tilt down. A bound that can also push
   up satisfies a canopy gap by pointing the camera into the trees, which is the
   measured failure this project is built to avoid.
3. A HELD SAMPLE IS NEVER GUESSED AT. Where no boundary was measured the path is
   carried, not driven to a target. A constraint that invents a row for a canopy
   frame is worse than no constraint, because it is confidently wrong.
"""

import numpy as np
import pytest

from steadycut.framing.path_optimizer import (
    SKY_BOUND, Constraints, solve, report,
)

N = 240
TIMES = np.arange(N, dtype=float) / 4.0        # 4 Hz, 60 s


def base_path():
    return -19.0 + 3.0 * np.sin(np.linspace(0, 6.0, N))


def test_unconstrained_is_the_identity():
    """No active constraint must reproduce the input exactly."""
    base = base_path()
    out = solve(Constraints(base=base, times=TIMES, iters=50))
    assert np.allclose(out, base, atol=1e-9)


def test_a_constant_cue_contributes_nothing():
    """A boundary already at target must not move the path, however many
    iterations run. This is the property that makes a degenerate cue harmless."""
    base = base_path()
    measured = np.full(N, 0.48)
    out = solve(Constraints(base=base, times=TIMES, boundary=measured, iters=200))
    assert np.allclose(out, base, atol=1e-6)


def test_a_misplaced_boundary_is_corrected_toward_target():
    base = base_path()
    boundary = np.full(N, 0.70)              # ground starts far too low
    out = solve(Constraints(base=base, times=TIMES, boundary=boundary,
                            iters=60, move_w=1.0, comp_w=4.0))
    after = boundary + (out - base) / 90.0
    assert np.nanmean(np.abs(after - 0.48)) < 0.02, after.mean()


def test_the_sign_of_the_correction():
    """A boundary too LOW in frame means little ground shows, so the view must
    tilt DOWN. Getting this backwards points the camera at the sky, which is the
    failure the dark-mass rule already caused once."""
    base = np.full(N, -19.0)
    out = solve(Constraints(base=base, times=TIMES,
                            boundary=np.full(N, 0.70), iters=40))
    assert out[0] < base[0], "should tilt down, not up"


def test_sky_bound_only_ever_tilts_down():
    base = np.full(N, -19.0)
    # a share that is over the bound at EVERY pitch, so the only escape is down
    share = lambda v: 0.9 if v > -45.0 else 0.0     # noqa: E731
    out = solve(Constraints(base=base, times=TIMES, sky_share=share, iters=20))
    assert out.max() <= base.max() + 1e-9, "bound pushed the view up"


def test_sky_bound_is_inactive_when_already_under():
    base = np.full(N, -19.0)
    out = solve(Constraints(base=base, times=TIMES,
                            sky_share=lambda v: 0.0, iters=20))
    assert np.allclose(out, base, atol=1e-9)


def test_a_held_sample_is_never_driven_to_a_target():
    """Canopy frames: the path is carried, not corrected. A constraint that
    invents a row for a frame with no boundary is confidently wrong."""
    base = base_path()
    boundary = np.full(N, 0.70)
    hold = np.zeros(N, dtype=bool)
    hold[100:140] = True
    out = solve(Constraints(base=base, times=TIMES, boundary=boundary,
                            hold=hold, iters=60))
    held_shift = np.abs(out[100:140] - base[100:140])
    free_shift = np.abs(out[:50] - base[:50])
    assert np.median(held_shift) < np.median(free_shift) * 0.5, (
        f"held {np.median(held_shift):.2f} vs free {np.median(free_shift):.2f}")


def test_smoothing_suppresses_a_head_reaction():
    """A sharp spike in the base path is the head-reaction, an uncorrected error
    source. The smoothness term is the only thing that can address it.

    Measured on the whole path, not at the spike: a single sample can be pulled
    by any local term, so asserting on that index alone does not show the term
    doing anything.
    """
    base = base_path()
    base[120] += 8.0
    out = solve(Constraints(base=base, times=TIMES, smooth_w=1.5, iters=300))
    before = np.abs(np.diff(base)).max()
    after = np.abs(np.diff(out)).max()
    assert after < before, f"largest step {after:.2f} vs {before:.2f}"
    # The movement is the spike being redistributed, so it must be LOCAL: the far
    # field of the path is untouched, and total movement cannot exceed the spike.
    far = np.abs(out - base)
    away = np.abs(np.arange(N) - 120) > 8
    assert far[away].max() < 0.1, f"far field moved {far[away].max():.3f}"
    assert far.sum() < abs(base[120] - base_path()[120]) + 0.5, far.sum()


def test_output_stays_in_bounds():
    base = np.full(N, 0.0)
    out = solve(Constraints(base=base, times=TIMES,
                            boundary=np.full(N, 0.99), iters=100))
    assert out.min() >= -60.0 and out.max() <= 60.0


def test_report_measures_the_truth_not_the_intent():
    """A report that cannot show a constraint failing is not a report."""
    base = base_path()
    boundary = np.full(N, 0.80)              # badly off target everywhere
    out = solve(Constraints(base=base, times=TIMES, boundary=boundary, iters=60))
    rep = report(Constraints(base=base, times=TIMES, boundary=boundary), out)
    assert rep["boundary_error_before"] > rep["boundary_error_after"]
    assert rep["moved_samples"] > 0
    assert rep["max_shift"] > 0


def test_empty_input_is_not_a_crash():
    out = solve(Constraints(base=np.array([]), times=np.array([])))
    assert len(out) == 0
