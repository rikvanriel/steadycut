"""MTB travel-holder: the opt-in policy for a camera that gets knocked over.

Same mount and anchor as mtb (helmet, bar in the bottom ~10%), but the
stabilisation trend is built in the VQF world frame instead of the
integrate frame: heading rides the production leash, pitch rides a slow
(0.02 Hz class) leash with an amplitude+duration holder, roll stays
production. A sustained world elevation above ~35 deg for ~10 s engages
the holder -- the measured knock signature (119 s at >30 deg, nothing
else on 30 min past 7 s) -- and the pitch freezes at last-good instead
of touring the canopy. Refusal anywhere in the world path falls back to
the legacy gaze path for that window, never to a degraded world frame.

OPT-IN until a real head-down window passes the eye check: select with
--framing mtb_travel. The default mtb policy is untouched.
"""
from __future__ import annotations

import math

from steadycut.framing.policies import FramingPolicy, register
from steadycut.framing.policies.mtb import (BAND, TARGET, dark_mass_edge,
                                             POLICY as _MTB)

POLICY = register(FramingPolicy(
    name="mtb_travel",
    mount=_MTB.mount,
    follows="travel",
    anchor=_MTB.anchor,
    occluders=_MTB.occluders,
    pitch=None,                     # must be measured per recording
    search=_MTB.search,
    yaw_corner_hz=None,             # heading rides the production leash
    yaw_adaptive_hz=None,
    pitch_tau_s=1.0 / (2.0 * math.pi * 0.02),  # slow holder leash on pitch
    corner_hz=0.15,
    hint=_MTB.hint,
    notes="Travel-holder variant of mtb: world-frame trend with a "
          "knock holder on pitch. Opt-in (--framing mtb_travel) until a "
          "real head-down window passes the eye check.",
    capabilities=_MTB.capabilities | {"travel_holder"},
))
