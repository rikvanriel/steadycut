"""Framing policies: the mtb policy resolves, an unknown name does not.

The policy layer exists so a second mount can be added as one file rather than
as edits scattered through the CLI and the path builder, so these tests pin the
things a new policy has to declare: whose orientation the IMU measures, which
orientation the framing follows, and how the pitch is derived.
"""
import numpy as np
import pytest

from steadycut.core.profiles import Profile
from steadycut.framing.ground_cue import GroundCue
from steadycut.framing.policies import POLICIES, policy_for, policy_names, register
from steadycut.framing.policies.ground import _cue_usable
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


def test_the_ground_policy_will_not_sweep_without_a_fitted_cue() -> None:
    """A ground cue is per recording, so its search is a factory.

    The regression this locks: before the CLI resolved factories, a missing cue
    would have had to be swept anyway, and a sweep with no target interpolates
    to an END OF THE LADDER and reports it as a usable pitch. Returning None is
    the only safe answer, and the CLI turning None into a refusal is what keeps
    "-5.0 degrees" from being presented as a measurement.
    """
    ground = policy_for("ground")
    assert ground is not None
    assert callable(ground.search), "ground must build its search per source"
    assert ground.search("no-such-recording-anywhere.insv") is None
    assert "needs_fitted_cue" in ground.capabilities
    # The fallback the hint points at must remain a fixed search, or the same
    # failure reaches the one policy that is supposed to always be available.
    mtb = policy_for("mtb")
    assert mtb is not None
    assert not callable(mtb.search)


def test_policy_names_are_listed_for_refusals() -> None:
    # Every registered policy, because the refusal that tells the rider what to
    # try instead is built from this list -- a policy that is not named here is
    # a policy nobody can discover from a failed run. Asserted by membership so
    # adding a third policy does not make this test the thing that has to be
    # rewritten, but the mtb name is kept explicit since it is the fallback the
    # hints point at.
    names = policy_names()
    assert "mtb" in names
    assert "ground" in names
    assert len(set(names)) == len(names), "a policy is registered twice"


def test_register_refuses_duplicates() -> None:
    with pytest.raises(ValueError):
        register(MTB)  # already registered by the package import


def test_the_measured_pitch_criterion_is_stated() -> None:
    """A policy must say what it is aiming at, in checkable words."""
    assert "90% of frame height" in MTB.anchor


def _profile(weights, err):
    cue = GroundCue(weights=np.asarray(weights, dtype=float),
                    mean=np.zeros(5), scale=np.ones(5))
    return Profile(pitch=-12.0, cue=cue, fitted_on={"held_out_error": err},
                   source_size=1, source_mtime=0, notes={})


def test_a_nan_error_cue_reads_as_missing() -> None:
    """The Oct-1 shape: 7 frames, NaN held-out error, stored anyway.

    A cue that predicted nothing finite has no target, and a sweep with no
    target brackets to a ladder end. It must read as missing so the CLI
    refuses instead.
    """
    p = _profile([0.9, -2.1, 0.8, 0.1, 5.1, 2.9], float("nan"))
    assert not _cue_usable(p)


def test_a_bias_only_cue_reads_as_missing() -> None:
    """All feature weights zero: the answer is the geometry prior.

    Whatever the picture shows, the boundary is the prior's -- so the sweep
    would bracket the prior, not the trail. Same refusal as missing.
    """
    p = _profile([0.0, 0.0, 0.0, -0.0, 0.0, -5.283], 0.02)
    assert not _cue_usable(p)


def test_a_fitted_cue_with_real_error_is_usable() -> None:
    """The 0519 shape: real weights, real held-out error.

    The guard must not eat honest cues: the loader prefers by content hash,
    so refusing here would strand every recording behind its own bad fit.
    """
    p = _profile([0.925, -2.131, 0.797, 0.107, 5.062, 2.909], 0.0197)
    assert _cue_usable(p)
