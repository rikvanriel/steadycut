"""Turn a measured boundary into a view pitch, with the sky bound on top.

The composition target and the offset are two different numbers and it is worth
keeping them apart. The offset is a MEASUREMENT: the bar sits 0.40 of frame
height (36 degrees at v_fov 90) below the ground/forest boundary, measured from
frames where both the boundary and the bike are visible. The target is a CHOICE:
where the boundary should sit in frame. Set the target to 0.48 and the measured
offset puts the bar at 0.88, near the bottom of frame -- which is what the mtb
policy has wanted all along, arrived at from the image rather than from the dark
mass that cannot be measured in canopy.

Where the boundary is not measurable the answer is HOLD, not guess: carry the
last corrected pitch forward and let the gyro stabilisation remove the rider's
motion across the gap. On the next measurable sample the correction re-anchors;
comparing the two ends of a held interval is what makes the bridge checkable.

The sky bound is applied last and only downward, so the composition correction
cannot buy a boundary it wants by pointing at the sky.

Pure functions: the caller supplies the sky share as a function of view pitch,
so nothing here needs the equirect or a renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET = 0.48            # where the boundary should sit, as a fraction of height
V_FOV = 90.0             # of the delivered frame, matching the observation view
BOUND = 0.25             # sky allowed in the top band
FLOOR = -45.0
SMOOTH_S = 1.0           # the correction is a per-recording measurement


@dataclass
class Correction:
    pitch: np.ndarray        # one value per sample
    held: np.ndarray         # True where no boundary was measurable
    acted: np.ndarray        # True where the sky bound moved the pitch

    @property
    def held_count(self) -> int:
        return int(self.held.sum())

    @property
    def acted_count(self) -> int:
        return int(self.acted.sum())


def correct(boundaries, base: float, share=None, target: float = TARGET,
            v_fov: float = V_FOV, bound: float = BOUND, floor: float = FLOOR,
            smooth_s: float = SMOOTH_S, fps: float = 1.0) -> Correction:
    """Per-sample view pitch from per-sample boundary measurements.

    `boundaries` is a per-column array per sample (NaN where unknown), the same
    shape `ground_cue.boundary` returns. `share` maps a view pitch to the sky
    share of the top band; omit it to skip the bound.
    """
    from steadycut.framing.sky_bound import bound_pitch

    raw, held = [], []
    last = base
    for b in boundaries:
        b = np.asarray(b, dtype=float)
        frac = float(np.nanmean(b)) if np.isfinite(b).any() else float("nan")
        if not np.isfinite(frac):
            raw.append(last)                 # hold: do not guess a row
            held.append(True)
        else:
            # Tilting down raises the ground in frame, so it LOWERS the fraction
            # where the boundary starts. Hence the sign: too low a fraction
            # means too much ground is showing, which asks for less pitch.
            raw.append(base + (target - frac) * v_fov)
            held.append(False)
        last = raw[-1]

    pitch = np.asarray(raw, dtype=float)
    if smooth_s and len(pitch) > 1:
        k = max(1, int(round(smooth_s * fps)))
        if k % 2 == 0:
            k += 1
        if k > 1 and k <= len(pitch):
            pitch = np.convolve(np.pad(pitch, k // 2, mode="edge"),
                                np.ones(k) / k, mode="valid")

    acted = np.zeros(len(pitch), dtype=bool)
    if share is not None:
        for i, p in enumerate(pitch):
            p2, moved = bound_pitch(share, float(p), bound=bound, floor=floor)
            if moved:
                pitch[i], acted[i] = p2, True
    return Correction(pitch=pitch, held=np.asarray(held), acted=acted)
