"""The validity gate must fire on real failure cases, not just pass its own tests.

A guard that never fires is the mirror of a detector that underreads, and equally
harmful: the suite goes green and the report claims a case is handled while the
failure stays exactly where it was. The degeneracy check built earlier in this
project fired on 0 of 4 good windows AND 0 of 4 broken ones while its unit tests
were green, so this module's tests are written to establish FIRING on the failure
modes, not merely that it does not crash on the good ones.

The numbers the assertions quote are the rider's own, from labelled real frames:
three frames called "upside down and sideways" and one "straight down, still
putting my helmet on" sat inside the good band on pitch, at roll 134 to 158
degrees, and a 45 degree roll limit caught 5 of the 6 not-riding frames.
"""
import numpy as np
import pytest

from steadycut.framing.validity import ROLL_LIMIT_DEG, assess, summary
from steadycut.stabilization.attitude_vqf import attitude

G = 9.80665
DT = 0.01


def _at(roll_deg, n=300, pitch_deg=0.0):
    """Raw IMU columns for a camera held at a known roll and pitch."""
    acc = np.zeros((n, 3))
    tr, tp = np.radians(roll_deg), np.radians(pitch_deg)
    # gravity on raw index 1, rolled onto index 2 and pitched onto index 0
    acc[:, 1] = G * np.cos(tr) * np.cos(tp)
    acc[:, 2] = G * np.sin(tr)
    acc[:, 0] = G * np.sin(tp)
    return attitude(np.zeros((n, 3)), acc, DT)


# --- the gate FIRES on the real failure classes ---

@pytest.mark.parametrize("roll_deg", [134.5, 140.0, 158.4])
def test_it_refuses_the_inverted_frames_the_rider_named(roll_deg):
    """Real values: 0519b 1298s, 0526a 4s, 0519b 1299s, all "upside down".

    These are the frames that PITCH cannot see -- their pitch sits inside the
    good band -- which is the entire reason this gate exists.
    """
    v = assess(_at(roll_deg))
    assert not v.valid.any(), f"roll {roll_deg} was accepted as a riding shot"
    assert "not level" in str(v.reason[0])


def test_it_accepts_a_level_riding_shot():
    v = assess(_at(6.0, pitch_deg=12.0))
    assert v.valid.all()
    assert all(r == "" for r in v.reason)


def test_a_barely_tilted_shot_is_still_delivered():
    """The limit must not fire early: 30 degrees is a real lean, not a fault."""
    assert assess(_at(30.0)).valid.all()


# --- the gate's FIRES WITH A REASON, because "refused" is not the deliverable ---

def test_every_refused_sample_carries_a_reason():
    v = assess(_at(150.0))
    assert (~v.valid).any()
    assert all(str(r) for r in v.reason[~v.valid])


def test_summary_reports_the_reasons_it_refused_for():
    s = summary(_at(150.0))
    assert s["refused"] == s["samples"]
    assert s["refused_fraction"] == 1.0
    assert any("not level" in k for k in s["reasons"])


# --- the gate says what it did NOT check ---

def test_it_declares_the_conditions_it_did_not_evaluate():
    """Speed and the visible boundary are NOT implemented, and saying so is part
    of the deliverable. A caller reading a clean summary must not conclude the
    0519b case is covered."""
    s = summary(_at(5.0))
    assert set(s["not_evaluated"]) == {"speed", "visible_boundary"}


# --- the control: a limit that never fires is a failure, so prove it discriminates ---

def test_a_limit_so_loose_it_never_fires_would_still_pass_a_smoke_test():
    """Why the firing tests above exist, stated as an executable claim.

    With a 200 degree limit every frame here is accepted -- including the camera
    at 158 degrees the rider called upside down. A suite that only checked "does
    not crash" would call that a pass. This asserts the MISS is visible so the
    distinction cannot be lost.
    """
    loose = assess(_at(158.4), roll_limit_deg=200.0)
    assert loose.valid.all(), "a 200 degree limit is the never-firing case"
    tight = assess(_at(158.4), roll_limit_deg=ROLL_LIMIT_DEG)
    assert not tight.valid.any(), "the shipped limit must catch the same frame"


def test_non_finite_attitude_is_refused_not_passed():
    v = assess(_at(5.0, n=200))
    broken = type(v)(
        valid=v.valid, reason=v.reason, roll=v.roll, pitch=v.pitch)
    object.__setattr__(broken, "roll",
                       np.where(np.arange(200) == 100, np.nan, broken.roll))
    w = assess(broken)
    assert not w.valid[100]
    assert "not-measurable" in str(w.reason[100])
