"""Framing policies: the mtb policy resolves, an unknown name does not.

The policy layer exists so a second mount can be added as one file rather than
as edits scattered through the CLI and the path builder, so these tests pin the
things a new policy has to declare: whose orientation the IMU measures, which
orientation the framing follows, and how the pitch is derived.
"""
import pytest

from steadycut.framing.policies import POLICIES, policy_for, policy_names, register
from steadycut.framing.policies.mtb import POLICY as MTB


def test_the_mtb_policy_is_registered_and_describes_its_mount() -> None:
    assert POLICIES["mtb"] is MTB
    # The IMU sits on the helmet, so it measures the HEAD -- which is both the
    # signal (where the rider looks) and the noise (bob, body english). That is
    # why this policy can follow the rider's own gaze with no estimator.
    assert MTB.mount == "helmet"
    assert MTB.follows == "gaze"
    assert "bar in the bottom" in MTB.anchor
    # The pitch is derived per recording by a criterion the policy supplies,
    # so the sweep stays mount-independent: no pitch transfers across rides.
    assert MTB.search is not None, "the pitch is measured per recording"
    assert MTB.search.target == 0.90
    assert callable(MTB.search.criterion)
    assert MTB.pitch is None, "no pitch transfers across recordings"
    assert "helmet" in " ".join(MTB.occluders)


def test_an_unknown_policy_returns_none() -> None:
    assert policy_for("no-such-mount") is None
    assert policy_for(None) is None


def test_policy_names_are_listed_for_refusals() -> None:
    assert policy_names() == ["mtb"]


def test_register_refuses_duplicates() -> None:
    with pytest.raises(ValueError):
        register(MTB)  # already registered by the package import


def test_the_measured_pitch_criterion_is_stated() -> None:
    """A policy must say what it is aiming at, in checkable words."""
    assert "90% of frame height" in MTB.anchor
