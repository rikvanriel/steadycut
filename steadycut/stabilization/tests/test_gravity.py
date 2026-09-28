"""The levelling filter must converge to a known tilt -- and diverge if sign-flipped.

`steadycut/stabilization/attitude.py` is the third gravity-fusing filter tried
here and it fails mechanically: its hard 20-degree innovation gate accepts
0.6-1.4 percent of samples on riding footage, so the accelerometer correction is
inert and what remains is a gyro-only propagation diverging 41-78 degrees from
the measured gravity. A hard gate rejects exactly the error that needs fixing.

The filter under test has NO innovation gate, in the shape the shipping tool
uses: a fixed multiplicative correction on the angular RATE, so a large error
always gets a large pull. The unit test below is the arithmetic check that the
shape converges, and it carries the control the previous attempts were missing.

The control is the point of this file. **A gravity filter with a flipped
correction sign diverges monotonically, and that is the only test that would
have caught the original fault** -- the previous attempt applied 90-180 degrees
instead of a few, and nothing in the suite could tell a sign error from a filter
that was merely too slow to converge. Here a sign flip is asserted to DIVERGE, so
a future edit that inverts the correction fails loudly instead of producing a
plausible plateau.

A synthetic fixture is the right instrument for the filter's ARITHMETIC -- it
converges, or it does not, and the answer is known by construction. It says
nothing about whether the cue's premise holds on real footage, which is why the
acceptance for this work is judged on RENDERED VIDEO, not here. The one real
property encoded below is the accelerometer's own fault: this body reads |a|
about 7 percent high, so a magnitude window stated against 1 g in absolute terms
rejects every sample, and the test asserts the window is relative instead.
"""
import numpy as np
import pytest

from steadycut.stabilization import gravity as GV

G = 9.81
WORLD_UP = np.array([0.0, 0.0, 1.0])


def _quat(axis, ang_rad):
    """Unit quaternion for a rotation of `ang_rad` about `axis`."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    s = np.sin(ang_rad / 2.0)
    return np.array([np.cos(ang_rad / 2.0), a[0] * s, a[1] * s, a[2] * s])


def _rot(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    t2 = 2.0 * np.cross(qv, v)
    return v + w * t2 + np.cross(qv, t2)


def synthetic(roll_deg=20.0, n=3000, fs=100.0, a_scale=1.0, hold=0.5,
              lat_g=0.0, lat_hz=0.4):
    """A known roll ramp, with the accelerometer reading the true gravity.

    Returns (t, gyro, accel, q_true). The camera rolls `roll_deg` about its
    optical axis over the first (1 - hold) of the window and then holds, so the
    tail carries a constant, known tilt that a correct filter must recover.

    `lat_g` adds sustained LATERAL acceleration along the body x axis, which is
    what leaning through a corner does to a real accelerometer: it tilts
    apparent gravity by 17-27 degrees and is the contaminant the magnitude
    window exists to reject. Without it the window cannot be tested at all,
    because a still body sits at one |a| and no window can reject anything --
    which is exactly why an earlier version of this fixture could not catch an
    absolute window.
    """
    t = np.arange(n) / fs
    ramp = np.clip(t / (t[-1] * (1.0 - hold)), 0.0, 1.0)
    ang = np.radians(roll_deg) * ramp

    # body frame: optical axis +x, camera up +y (gravity sits on the up axis,
    # which is the measured fact for this body). Roll is about +x.
    q_true = np.empty((n, 4))
    for i in range(n):
        q_true[i] = _quat([1.0, 0.0, 0.0], ang[i])

    # gyro is the body rate about x, in deg/s (this project's units)
    rate = np.gradient(ang, t)
    gyro = np.zeros((n, 3))
    gyro[:, 0] = np.degrees(rate)

    # accelerometer measures specific force: the world up direction expressed
    # in the body frame, plus any linear acceleration, scaled by g and by the
    # body's own gain error.
    lat = lat_g * G * np.sin(2 * np.pi * lat_hz * t)
    accel = np.empty((n, 3))
    for i in range(n):
        up_body = _rot(_conj(q_true[i]), WORLD_UP)
        accel[i] = (up_body + np.array([lat[i], 0.0, 0.0])) * G * a_scale
    return t, gyro, accel, q_true


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _tilt_error(q_est, q_true, t, frac=0.25):
    """Tail error in degrees between the estimate and the truth, as TILT only.

    Compares the WORLD-UP DIRECTION EXPRESSED IN THE BODY FRAME for the two
    attitudes, and nothing else. That is precisely the quantity gravity can
    observe: it is invariant to yaw, because a yaw about the world up leaves it
    unchanged, so a filter cannot score itself on a degree of freedom it has no
    information about.

    The first version of this compared `_rot(q, world_up)` instead, which
    applies the body-to-world rotation to a WORLD vector. That quantity is
    identical for the true and the estimated attitude whenever the estimate is
    right, and it is not the camera's tilt in any case -- which made the filter
    look inert (a 40.00 deg error on a 40 deg ramp) when it was tracking to four
    decimals. The filter was right and the test was wrong.
    """
    n = len(t)
    lo = int(n * (1.0 - frac))
    errs = []
    for i in range(lo, n):
        a = _rot(_conj(q_true[i]), WORLD_UP)     # world up, in the true body frame
        b = _rot(_conj(q_est[i]), WORLD_UP)      # world up, in the estimated frame
        c = np.clip(float(a @ b), -1.0, 1.0)
        errs.append(np.degrees(np.arccos(c)))
    return float(np.median(errs))


def test_the_filter_recovers_a_known_tilt() -> None:
    """A 20 degree roll must be recovered to within 2 degrees in the tail.

    The tail is the point: a filter that starts at identity has all the time in
    the world to converge, and only the settled answer is worth anything.
    """
    t, gyro, accel, q_true = synthetic(roll_deg=20.0)
    q = GV.level_attitude(t, gyro, accel, tau_s=5.0)
    err = _tilt_error(q, q_true, t)
    assert err < 2.0, f"tail tilt error {err:.2f} deg, criterion is 2 deg"


def test_a_flipped_correction_sign_diverges() -> None:
    """THE CONTROL. A sign error must be loud, not a plausible plateau.

    This is the test the previous attempt lacked. An inverted gravity correction
    pushes the attitude AWAY from level, and the divergence grows without bound
    because the pull is proportional to the error and the error is what the pull
    increases.
    """
    t, gyro, accel, q_true = synthetic(roll_deg=20.0)
    q = GV.level_attitude(t, gyro, accel, tau_s=5.0, sign=-1.0)
    err = _tilt_error(q, q_true, t)
    assert err > 10.0, (
        f"a sign-flipped correction settled at {err:.2f} deg error; if this "
        f"fails the filter is too weak to detect its own sign error, which is "
        f"the fault this test exists to catch")


def test_the_magnitude_window_is_invariant_to_the_bodies_gain_error() -> None:
    """The window must test for ACCELERATION, not for a gravity calibration error.

    The window must reject the CORNERING and accept the calm, and must do so
    identically on a body that reads 7 percent high -- because a constant gain
    error is not motion. An absolute window keyed to 1 g is neither, and fails
    this at ANY gain error once the signal is realistic.

    The fixture therefore carries 0.4 g of lateral acceleration. Without it
    nothing can be rejected at all and both windows pass 100 percent, which is
    why the first two versions of this test could not catch the regression they
    were written for.
    """
    _, _, accel_a, _ = synthetic(roll_deg=20.0, a_scale=1.00, lat_g=0.4)
    _, _, accel_b, _ = synthetic(roll_deg=20.0, a_scale=1.07, lat_g=0.4)
    keep_a = GV.magnitude_window(accel_a)
    keep_b = GV.magnitude_window(accel_b)
    assert 0.05 < keep_a.mean() < 0.95, (
        f"the window accepted {keep_a.mean() * 100:.0f} percent of a window "
        f"that includes 0.4 g of cornering; a window that rejects everything is "
        f"the inert-correction failure and one that rejects nothing is not a "
        f"window at all")
    assert np.array_equal(keep_a, keep_b), (
        "the accepted set changed when only the body's gain changed, so the "
        "window is reading the calibration rather than the acceleration")


def test_a_body_reading_7_percent_high_still_converges() -> None:
    """The end-to-end consequence of the window being relative.

    Kept separately from the window's own test because this is the property the
    render depends on: a calibration error must not cost the filter its gravity
    reference, or the horizon simply is not levelled.
    """
    t, gyro, accel, q_true = synthetic(roll_deg=20.0, a_scale=1.07)
    q = GV.level_attitude(t, gyro, accel, tau_s=5.0)
    err = _tilt_error(q, q_true, t)
    assert err < 2.0, (
        f"a 7 percent accelerometer gain error broke the filter: tail error "
        f"{err:.2f} deg. The magnitude window must be relative.")


def test_there_is_no_innovation_gate_because_a_gate_cannot_re_anchor() -> None:
    """A large error must get a large pull, which is precisely what a gate forbids.

    The previous filter's 20-degree innovation gate is why it could never
    re-anchor: the samples that would fix the error are the ones it rejects. So
    the test starts the filter 60 degrees out. With a gate at any width below
    60 degrees the correction can never apply and the attitude never moves; with
    no gate the same run converges.

    The earlier version of this test ramped from 0 to 40 degrees starting at
    identity, so the innovation never exceeded the gate and the mutation that
    reintroduced the gate PASSED. A guard that a regression walks straight
    through is decoration.
    """
    t, gyro, accel, q_true = synthetic(roll_deg=20.0, n=6000)
    # start 60 degrees away from level about the optical axis
    q0 = np.array([np.cos(np.radians(60) / 2), np.sin(np.radians(60) / 2), 0.0, 0.0])
    q = GV.level_attitude(t, gyro, accel, tau_s=5.0, q0=q0)
    err = _tilt_error(q, q_true, t)
    assert err < 2.0, (
        f"the filter could not re-anchor from 60 degrees out: tail error "
        f"{err:.2f} deg. A correction that shrinks with the error it is fixing "
        f"is an innovation gate wearing another name.")


def test_a_long_tau_is_accepted_and_a_slow_one_is_slower_to_settle() -> None:
    """Levelling wants a LONG tau; the plan's sweep is 40/20/10/5/1.7 s.

    Gyroflow's 1.7 s is tuned for a 5 s bounce, not for a 30 s window whose
    contamination must average out, so the parameter has to span that range
    rather than being pinned to the shipping default.
    """
    t, gyro, accel, q_true = synthetic(roll_deg=20.0, n=6000)
    for tau in (40.0, 20.0, 10.0, 5.0, 1.7):
        q = GV.level_attitude(t, gyro, accel, tau_s=tau)
        err = _tilt_error(q, q_true, t)
        assert err < 4.0, f"tau {tau} s failed to settle: {err:.2f} deg"
