"""Framing policies: what to point the virtual camera at, and why.

A camera profile says what the BODY can do (lens geometry, fov ceilings,
capabilities). A framing policy says what the SHOT should look like: where the
subject sits in frame, which orientation the framing follows, and what the
IMU's axes actually mean for this mount. The two are orthogonal -- the same
camera on a helmet and on a handlebar needs the same lens arithmetic and a
different framing policy.

**Mount geometry decides everything else**, so it is the first field of a
policy. The IMU measures the orientation of whatever it is bolted to:

  * helmet   -- the IMU measures the RIDER'S HEAD, which leads the turn and
                wonders off the line. That orientation is signal (where the
                rider looks) as well as noise (bob, body english).
  * bar/bike -- the IMU measures the BIKE, which is rigid with the trail: it
                does not lead the turn, and its roll is lean, not gaze.
  * shirt    -- the IMU measures the TORSO: between the two, and occluded by
                the rider's own arms and shoulders rather than by a helmet.
  * fpv      -- the IMU measures the AIRFRAME: its rotation is the flying, and
                the whole "follow the operator's intent" idea changes meaning.

Those differences are not cosmetic: a correction tuned to cancel head bob
cancels the wrong signal on a bar mount, and a framing criterion derived from
where a helmet sits in frame means nothing from a shirt.

Each policy is ONE file here (mtb.py, later road_bar.py, shirt_bike.py,
fpv_ski.py, ...) registering a FramingPolicy. A policy carries:

  mount        what the camera is attached to, and whose orientation the IMU
               therefore measures
  follows      whose orientation the framing follows: the head ("gaze"), the
               direction of travel ("travel"), or nothing ("fixed")
  pitch        a fixed framing pitch, used when measuring is not possible
  measure      the per-recording procedure that derives the pitch, or None
  anchor       the framing criterion, in words a human can check
  occluders    what the mount guarantees will be in frame, so the crop can be
               biased away from it instead of spending pixels on it
  hint         what to tell the user when the measurement is unusable

An open question the existing footage cannot answer, to be settled by the new
videos rather than by guessing: whether "intentional motion" (what the framing
follows, i.e. the corner's other side) means the same thing on a bar mount --
where the bike's roll is the trail leaning, not a gesture -- as it does on a
helmet, where it is the rider's head leading the turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class FramingPolicy:
    """One way of framing one mount: the subject anchor and how to hold it."""

    name: str
    mount: str                      # helmet | bar | shirt | fpv
    follows: str                    # gaze | travel | fixed
    anchor: str                     # the criterion, in checkable words
    occluders: tuple[str, ...] = ()
    pitch: float | None = None      # fixed fallback; None = must be measured
    measure: Callable | None = None  # (source, start, workdir) -> (pitch, info)
    hint: str = "re-try with --pitch set by eye"
    notes: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)


POLICIES: dict[str, FramingPolicy] = {}


def register(policy: FramingPolicy) -> FramingPolicy:
    """Register a policy from its own file; returns it for re-export."""
    if policy.name in POLICIES:
        raise ValueError(f"duplicate framing policy {policy.name}")
    POLICIES[policy.name] = policy
    return policy


def policy_for(name: str | None) -> FramingPolicy | None:
    """The named policy, or None when there is no such policy."""
    if not name:
        return None
    return POLICIES.get(name)


def policy_names() -> list[str]:
    """Every registered policy name, sorted -- for refusal messages."""
    return sorted(POLICIES)

# Importing the policy files registers them, exactly as the camera package does
# for its profiles: anything that resolves a policy imports this package, so
# registration is guaranteed before lookup.
from steadycut.framing.policies import mtb  # noqa: E402,F401  (registers "mtb")
