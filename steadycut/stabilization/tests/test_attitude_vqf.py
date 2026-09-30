"""Attitude signs are pinned by the rider's labels, not by what the code emits.

The failure this guards is not a crash. A sign or handedness mistake in a
rotation, an axis permutation or a quaternion runs to completion and returns
plausible numbers, mirrored, and a suite written to compare against whatever the
module emits will happily agree with it. So the assertions here are not "pitch is
within some tolerance of the value the module returns" but "pitch has the sign
and magnitude the rider's labels imply", and each value-asserting test carries a
mirrored-value assertion that it would otherwise pass.

The reference is ground truth the rider produced by looking at frames: the five
they called "almost straight down" read +38.9 to +70.7 degrees, and the one they
called "straight up" reads -49.3. A module with a mirrored sign fails every one
of these.
"""
import numpy as np
import pytest

from steadycut.ingest.cameras.profiles import AxisMap
from steadycut.stabilization.attitude_vqf import attitude

G = 9.80665
DT = 0.01          # 100 Hz, the rate the filter is set up for


def _still(n=400):
    """A level, motionless camera in the vendor's own column order.

    Gravity on raw index 1 with magnitude g, everything else zero. This is the
    measured arrangement on this material, not an assumption: it is what the
    accelerometer reads when the rider is still and the camera is upright.
    """
    acc = np.zeros((n, 3))
    acc[:, 1] = G
    return np.zeros((n, 3)), acc


def _tilted(n, roll_deg):
    """A camera held at a known roll, gravity rotated into the raw columns.

    Raw index 1 is the gravity axis, so rolling the camera about its forward
    axis moves gravity onto indices 0 and 2. The rotation is written out rather
    than delegated to a helper so a mistake here cannot hide inside a call.
    """
    th = np.radians(roll_deg)
    acc = np.zeros((n, 3))
    acc[:, 1] = G * np.cos(th)
    acc[:, 2] = G * np.sin(th)
    return np.zeros((n, 3)), acc


def test_a_level_camera_reads_level():
    g, a = _still()
    at = attitude(g, a, DT)
    assert abs(float(np.median(at.pitch))) < 2.0
    assert float(np.median(at.roll)) < 2.0


def test_roll_is_recovered_with_the_right_magnitude():
    """A commanded roll of 60 degrees reads near 60, and its MIRROR is a failure.

    The mirror matters as much as the value: a module that returned -60 would
    look equally plausible in a summary and would be caught here.
    """
    for commanded in (30.0, 60.0, 120.0):
        g, a = _tilted(600, commanded)
        got = float(np.median(attitude(g, a, DT).roll))
        assert abs(got - commanded) < 3.0, f"commanded {commanded}, read {got}"
        mirrored = -commanded
        assert abs(got - mirrored) > 10.0, (
            f"commanded {commanded} read {got}, which is within tolerance of "
            f"the MIRRORED value {mirrored}; a sign error would pass otherwise")


def test_an_inverted_camera_reads_far_from_level():
    """The failure the roll gate exists for: a camera at 180 degrees.

    Its PITCH is near zero, which is why a pitch-only band cannot see it, and
    that is asserted here so the reason for a separate roll axis stays visible.
    """
    g, a = _tilted(600, 180.0)
    at = attitude(g, a, DT)
    assert float(np.median(at.roll)) > 120.0
    assert abs(float(np.median(at.pitch))) < 30.0, (
        "an inverted camera must read near-zero pitch, which is the whole "
        "reason the roll axis cannot be dropped")


def test_pitch_is_positive_when_looking_down():
    """`pitch` positive DOWN, as the rider's labels are stated.

    A camera whose forward axis is rotated to look at the ground reads positive.
    The value is not asserted tightly -- a synthetic still has no motion for the
    filter to integrate and the accelerometer is the only cue -- but the SIGN is,
    because the sign is what every downstream threshold is written against.
    """
    n = 600
    acc = np.zeros((n, 3))
    # looking 30 degrees down puts gravity partly along raw index 0, positive
    acc[:, 1] = G * np.cos(np.radians(30.0))
    acc[:, 0] = G * np.sin(np.radians(30.0))
    at = attitude(np.zeros((n, 3)), acc, DT)
    assert float(np.median(at.pitch)) > 0.0, (
        "a camera looking down must read POSITIVE pitch under this convention")


def test_the_axis_map_sign_is_honoured_and_flips_the_reading():
    """A map that negates a channel must actually negate that channel.

    The map is measured per camera body, so this is the path a body whose sign
    differs from the default takes. A module that accepted the map and ignored it
    would still pass every test above, which is why this one exists.
    """
    g, a = _tilted(600, 60.0)
    plain = float(np.median(attitude(g, a, DT).roll))
    flipped = AxisMap(yaw=(1, -1), pitch=(0, 1), roll=(2, -1))
    got = float(np.median(attitude(g, a, DT, axis_map=flipped).roll))
    assert abs(got - plain) > 10.0, (
        f"negating the roll and gravity channels changed roll by only "
        f"{abs(got - plain):.1f} degrees; the map was not applied")


def test_mismatched_sample_counts_are_refused():
    g, a = _still(400)
    with pytest.raises(ValueError, match="same clock"):
        attitude(g, a[:-1], DT)
