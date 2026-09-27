"""Framing as a constrained path optimisation, per Tang/Wang/Liu (TOG 2018).

The scalar pitch this project has been chasing -- a per-recording constant, or a
slow correction to it -- cannot express the cases that keep going wrong: a
canopy gap with no boundary, a duck that puts the bike above the axis, a
head-reaction the sky bound cannot touch because it is one-sided. The 360
literature's answer is to stop solving for a scalar and solve for a PATH under
constraints, so that an infeasible case degrades to the minimum violation rather
than to nonsense.

The constraints here are the ones already measured on this footage, as look-at
targets rather than as objectives to chase:

  direction   the rider's own forward-motion direction (from the existing path)
  boundary    the ground/forest boundary at 0.48 of frame height
  bar         the bike's bar at 0.90, which with the measured 0.40 offset is
              the same statement about the same surface
  sky         at most 0.25 blue in the top band -- ONE-SIDED, and the reason is
              measured: a bound that can also push up satisfies a canopy gap by
              pointing the camera into the trees

The objective minimises total movement from the existing path, subject to those.
That ordering matters: with no active constraint the answer is exactly the
existing path, so this can only improve on it, and the unconstrained case is the
identity. A cue that turns out to be a per-recording constant (as the boundary
measurement now looks) contributes nothing rather than dragging the path.

Solved by projected gradient descent on a per-sample pitch sequence, with the
sky bound applied as a projection because it is genuinely one-sided, and the
composition targets as projections too -- a pitch that puts the boundary at 0.48
is reachable by rotation alone, and projecting onto it is exact rather than a
gradient step toward it.

Smoothing comes last and is a real term, not a tie-break: the head-reaction is
an uncorrected error source precisely because nothing penalises rapid rotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Measured on this footage, not assumed.
TARGET_BOUNDARY = 0.48      # ground/forest line, fraction of frame height
BAR_OFFSET = 0.40           # bar sits this far below the boundary
V_FOV = 90.0                # of the delivered 4:3 frame
SKY_BOUND = 0.25            # blue allowed in the top 30% band
FLOOR, CEIL = -60.0, 60.0
TRUST_BAND = 0.12           # |measured - target| below this is already fine


@dataclass
class Constraints:
    """What the path must satisfy. Each may be None, meaning 'no opinion'."""

    base: np.ndarray                       # the existing path, per sample
    times: np.ndarray
    boundary: np.ndarray | None = None     # measured boundary fraction
    sky_share: Callable[[float], float] | None = None   # pitch -> share, or None
    bar_top: np.ndarray | None = None      # measured bar row fraction
    hold: np.ndarray | None = None         # True where nothing was measurable
    smooth_hz: float = 0.5                 # kernel length, if smoothing is asked
    smooth_w: float = 0.0                  # OFF by default: see below
    move_w: float = 1.0                    # weight on staying near the base
    comp_w: float = 2.0                    # weight on hitting the composition
    target: float = TARGET_BOUNDARY
    iters: int = 400
    lr: float = 0.35
    report: dict = field(default_factory=dict)


def _project_composition(pitch, measured, target, v_fov=V_FOV):
    """Put `measured` at `target` by rotating, when measured is finite.

    THE SIGN, carefully, because it was got wrong twice in review. Tilting the
    view DOWN raises the ground in frame, so the fraction where ground starts
    DECREASES. A boundary measured too LOW in frame (large fraction, too little
    ground showing) therefore needs a DOWNWARD pitch to reach a smaller target:

        out = pitch - (measured - target) * v_fov

    and the inverse, used by report(), is `implied = measured + delta / v_fov`.
    An earlier version of report() used a minus there and so reported every
    correction as making things worse, which sent the projection's sign the
    wrong way. The projection itself was right; the measurement of it was not.
    """
    out = np.array(pitch, dtype=float)
    m = np.isfinite(measured)
    if not m.any():
        return out
    dev = np.where(m, measured - target, 0.0)
    big = m & (np.abs(dev) > TRUST_BAND)
    out[big] = out[big] - dev[big] * v_fov
    return out


def _project_sky(pitch, share, bound=SKY_BOUND, floor=FLOOR, iters=24):
    """One-sided: tilt DOWN only, never up. The asymmetry is measured.

    A bound that CANNOT BE MET is declined, not enforced. Found on real footage:
    a share function reading ~0.93 against a 0.25 bound sent every sample to the
    floor, so the path came back constant -- the gyro path discarded, no
    stabilisation at all -- while the sky share still improved and every metric
    that counts improvement reported success.

    The check is therefore whether the bound is reachable AT ALL inside
    [floor, v]: if even the floor is over the bound, no downward pitch can satisfy
    it, and running the search to the floor anyway just destroys the framing
    without satisfying anything. Those samples are reported so a caller can see
    the bound was skipped rather than met.
    """
    if share is None:
        return (np.array(pitch, dtype=float), np.zeros(len(pitch), dtype=bool),
                np.zeros(len(pitch), dtype=bool))
    out = np.array(pitch, dtype=float)
    acted = np.zeros(len(pitch), dtype=bool)
    unreachable = np.zeros(len(pitch), dtype=bool)
    for i in range(len(out)):
        v = out[i]
        if share(v) <= bound:
            continue
        if share(floor) > bound:
            # no pitch in the reachable range satisfies this
            unreachable[i] = True
            continue
        lo, hi = floor, v
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if share(mid) > bound:
                hi = mid
            else:
                lo = mid
        out[i] = 0.5 * (lo + hi)
        acted[i] = True
    return out, acted, unreachable


def solve(c: Constraints) -> np.ndarray:
    """Minimise movement + roughness subject to the constraints.

    The diagnostics (whether the bound was reachable, how many samples it was
    skipped on) are recorded on `c.report`, which `report()` reads. They are
    also on the Constraints object rather than returned, because a caller who
    re-constructs an identical Constraints to ask for the report would otherwise
    get an empty dict and conclude the bound was met. `report()` warns when it
    has no solve() diagnostics to read.

    Composition is an objective pull, not a projection: the observation view is
    rendered at a fixed orientation, so a measured boundary does not move when
    the delivered pitch does. Projecting it every iteration fought the other
    terms and converged to a compromise satisfying none of them.

    The sky bound IS a projection, because it is one-sided by measurement rather
    than by preference, and enforcing it as a soft term would let a sample drift
    above the bound and stay there.

    Smoothing is OFF unless asked for (`smooth_w=0`). It was on by default, and
    that broke the one property the formulation rests on: with no active
    constraint the answer must be the input path exactly, so that the optimiser
    can only ever improve on it. A default term that moves the path when nothing
    is wrong is a term that has to be switched off to be trusted.

    Order within an iteration: pull toward the measured composition, then toward
    the base path where nothing was measured, then toward the smoothed self if
    smoothing was asked for, then enforce the bound.
    """
    p = np.array(c.base, dtype=float)
    n = len(p)
    if n == 0:
        return p
    dt = float(np.median(np.diff(c.times))) if n > 1 else 1.0
    smoothness = None
    if c.smooth_hz > 0 and n > 2:
        tau = 1.0 / (2.0 * np.pi * c.smooth_hz)
        # A length-1 box filter is the IDENTITY, and round(tau/dt) is 1 for any
        # smooth_hz below ~1 Hz at 4 Hz sampling -- so the first version of this
        # term computed a kernel of one sample and smoothed nothing, which is why
        # the head-reaction test could not fail. A single-sample spike needs a
        # window spanning several samples to be attenuated at all.
        k = max(3, int(round(tau / dt)))
        k = min(k, n if n % 2 else n - 1)
        if k >= 3:
            ker = np.ones(k) / k
            pad = k // 2
            xp = np.pad(p, pad, mode="edge")
            smoothed = np.convolve(xp, ker, mode="valid")[:n]
            if np.size(smoothed) == n:
                smoothness = smoothed

    # The observation view is rendered at a FIXED orientation, so a measured
    # boundary does not move when the delivered pitch changes: it is a property
    # of the scene, not of the framing. That makes the composition term an
    # OBJECTIVE (a pull toward the target), not a projection (a hard set).
    # Projecting it every iteration fought the other two terms and converged to
    # a compromise satisfying none of them; the two earlier attempts made that
    # mistake in opposite directions.
    #
    # So: accumulate a desired pitch from whatever is measurable, pull toward it
    # with a weight, pull toward the base path, pull toward its own smoothed
    # self, and enforce the sky bound as the one hard projection -- which it
    # genuinely is, because it is one-sided by measurement, not by preference.
    desired = np.array(c.base, dtype=float)
    weight = np.zeros(n)
    if c.boundary is not None:
        m = np.isfinite(c.boundary)
        if c.hold is not None:
            m &= ~c.hold            # hold: do not guess a row
        if m.any():
            desired[m] = _project_composition(c.base[m], c.boundary[m], c.target)
            weight[m] = 1.0
    if c.bar_top is not None:
        bm = np.isfinite(c.bar_top)
        if bm.any():
            b_target = c.target - BAR_OFFSET
            want = _project_composition(c.base[bm], c.bar_top[bm], b_target)
            if weight[bm].any():
                # two statements of one surface: average them rather than let the
                # second overwrite the first
                desired[bm] = (desired[bm] + want) / 2.0
            else:
                desired[bm] = want
                weight[bm] = 1.0

    # Where something was MEASURED, the measurement is the answer and the base
    # path is not consulted at all. Where nothing was measured, the base path IS
    # the answer. Letting move_w pull toward base where a measurement exists made
    # the solve settle at 2/3 of the needed correction, because the two terms
    # balanced rather than one deferring to the other.
    seen = weight > 0
    if c.smooth_w <= 0.0:
        smoothness = None
    unreachable_total = 0
    for _ in range(c.iters):
        p = p - c.lr * c.comp_w * weight * (p - desired)
        if (~seen).any():
            p = p - c.lr * c.move_w * (~seen) * (p - c.base)
        if smoothness is not None:
            p = p - c.lr * c.smooth_w * (p - smoothness)
        p, _, unreach = _project_sky(p, c.sky_share)
        unreachable_total = max(unreachable_total, int(unreach.sum()))
        p = np.clip(p, FLOOR, CEIL)
    c.report["unsatisfiable_samples"] = unreachable_total
    c.report["bound_satisfied"] = unreachable_total == 0
    return p


def report(c: Constraints, solved: np.ndarray) -> dict:
    """What moved, and how much each constraint was binding."""
    base = np.asarray(c.base, dtype=float)
    out = dict(c.report) if c.report else {}
    if not c.report:
        # A report built from a Constraints that solve() never saw carries no
        # diagnostics, and a caller reading an absent key would conclude the
        # bound was met. Say so rather than let it pass.
        out["solved"] = False
    out.update({
        "samples": int(len(solved)),
        "moved_samples": int(np.sum(np.abs(solved - base) > 0.05)),
        "max_shift": float(np.max(np.abs(solved - base))) if len(solved) else 0.0,
        "rms_shift": float(np.sqrt(np.mean((solved - base) ** 2)))
        if len(solved) else 0.0,
    })
    if len(solved) > 2:
        r_base = np.abs(np.diff(base)).mean()
        r_solved = np.abs(np.diff(solved)).mean()
        out["mean_step_before"] = float(r_base)
        out["mean_step_after"] = float(r_solved)
    if c.boundary is not None:
        m = np.isfinite(c.boundary) & ~np.asarray(c.hold, dtype=bool)
        if m.any():
            out["boundary_error_before"] = float(
                np.nanmean(np.abs(c.boundary[m] - c.target)))
            # inverse of the projection: a delta of -dev*v_fov moves the implied
            # boundary by -dev, so implied = measured + delta/v_fov
            after = c.boundary[m] + (solved[m] - base[m]) / V_FOV
            out["boundary_error_after"] = float(np.nanmean(np.abs(after - c.target)))
    if c.sky_share is not None:
        sb = np.array([c.sky_share(v) for v in base])
        sa = np.array([c.sky_share(v) for v in solved])
        out["frames_over_sky_bound_before"] = int(np.sum(sb > SKY_BOUND))
        out["frames_over_sky_bound_after"] = int(np.sum(sa > SKY_BOUND))
    return out
