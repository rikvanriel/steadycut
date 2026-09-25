"""Camera profiles: the shared dataclasses and the model-name registry.

Each supported camera model defines a profile in its own file next to this one
(e.g. insta360_x4.py) and registers it here. This module holds ONLY the shared
types and the lookup.

The split matters: a profile carries what is MODEL-WIDE and known before any
footage is seen -- lens geometry, projection, stack order, capabilities, field
of view ceilings. What is MEASURED -- the gyro axis map of a particular body,
and anything else derived from its own recordings -- is stored per body in the
user's config file, not here (see store.py and registry.axis_map_for). So a
profile arrives with `axis_map=None` and the registry attaches the measured map
when the body is known; a caller that needs a map and has none must measure,
never assume.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxisMap:
    """Where each correction channel lives in the raw gyro columns.

    Per channel: (gyro column index, sign). Both are measured on the body from
    its own footage (steadycut.calibration.discover), and the result is stored
    in the user's camera registry. A mapping that reads correctly on paper can
    be exactly backwards, which is why this is never copied between models or
    guessed from a datasheet.
    """

    yaw: tuple[int, int]
    pitch: tuple[int, int]
    roll: tuple[int, int]


@dataclass(frozen=True)
class LensGeometry:
    """What the renderer needs to know about the lens arrangement.

    projection: the ffmpeg v360 input name (dfisheye, fisheye, rectilinear).
    lens_fov_deg: the full field of view of ONE lens. The renderer derives
    v360's id_fov from it in one place, including the sqrt(2) diagonal
    convention for a circular fisheye inscribed in a square frame.
    stack_order: which video track feeds which side of the hstack for a
    dual-lens body. The wrong order rotates the stitched sphere by 180
    degrees, which reads as a framing problem (the X4 lesson).
    """

    projection: str
    lenses: int
    lens_fov_deg: float
    stack_order: str
    travel_yaw: float = 0.0


@dataclass(frozen=True)
class CameraProfile:
    """Model-wide conventions for one camera model.

    Measured constants that belong to ONE BODY (the gyro axis map, mounting
    rotation, per-unit lens deltas) do NOT live here: they are stored per
    serial in the user's config and attached by the registry. `axis_map` here
    is therefore normally None, and a caller holding a profile with None must
    measure one rather than fall back to another camera's.
    """

    model: str
    telemetry_format: str            # insv_trailer | gpmf | dji | external | none
    axis_map: AxisMap | None         # None = not measured for this body: measure
    lens: LensGeometry
    imu_rate_nominal_hz: float | None
    max_useful_hfov: dict[str, float]   # per aspect, at the measured smear ceiling
    capabilities: frozenset[str] = frozenset()
    prestabilized_default: bool = False  # in-camera stabilisation baked into pixels
    notes: str = ""


MODEL_PROFILES: dict[str, CameraProfile] = {}


def register(profile: CameraProfile) -> CameraProfile:
    """Register a profile from its camera file; returns it for re-export."""
    if profile.model in MODEL_PROFILES:
        raise ValueError(f"duplicate profile for {profile.model}")
    MODEL_PROFILES[profile.model] = profile
    return profile


def profile_for(model: str | None) -> CameraProfile | None:
    """The profile for a model name, or None when we have none for it."""
    if not model:
        return None
    return MODEL_PROFILES.get(model)
