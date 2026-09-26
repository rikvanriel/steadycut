"""Per-recording framing profile: the measured pitch and the fitted cue.

Both of these are properties of ONE recording, and the reason is the same for
each. The pitch is a per-recording constant because the bar's elevation is
posture-dependent (44.5 degrees below the horizon at one moment, 38.6 at another,
on the same ride), so a number that holds one recording puts the bar out of frame
on the next. The cue is fitted from a handful of labelled frames of THIS
recording, so it learns this camera's mount and this ride's light.

Neither belongs in the source code, and neither is a user preference: they are
measurements of a clip, so they live beside the measurements -- under
$XDG_CONFIG_HOME/steadycut/profiles/, written when someone fits and read when the
policy renders. A missing profile is not an error. It means nobody has measured
this clip yet, and the policy falls back to the pitch search it has always used.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TARGET = 0.48


def config_dir() -> Path:
    """$XDG_CONFIG_HOME/steadycut, or ~/.config/steadycut."""
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "steadycut"


def key_for(source) -> str:
    """A stable key for a recording: its resolved path, digested.

    The path rather than the file's name alone, so two clips with the same name
    from different campaigns cannot collide, and so a profile can be found again
    without the file being loaded.
    """
    resolved = str(Path(source).expanduser().resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:16]


@dataclass
class Profile:
    pitch: float | None = None          # measured per recording, never carried
    target: float = TARGET              # where the boundary should sit in frame
    cue: Any = None                     # a fitted GroundCue, or None
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        cue = None
        if self.cue is not None:
            cue = {
                "weights": np.asarray(self.cue.weights).tolist(),
                "mean": np.asarray(self.cue.mean).tolist(),
                "scale": np.asarray(self.cue.scale).tolist(),
            }
        return {"version": 1, "pitch": self.pitch, "target": self.target,
                "cue": cue, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        cue = None
        if d.get("cue"):
            from steadycut.framing.ground_cue import GroundCue

            c = d["cue"]
            cue = GroundCue(weights=np.asarray(c["weights"], dtype=float),
                            mean=np.asarray(c["mean"], dtype=float),
                            scale=np.asarray(c["scale"], dtype=float))
        return cls(pitch=d.get("pitch"), target=d.get("target", TARGET),
                   cue=cue, notes=d.get("notes", {}))


def save(key: str, profile: Profile, directory: Path | None = None) -> Path:
    directory = Path(directory) if directory else config_dir() / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    dst = directory / f"{key}.json"
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile.to_dict(), indent=2))
    tmp.replace(dst)                   # never leave a half-written profile
    return dst


def load(key: str, directory: Path | None = None) -> Profile | None:
    directory = Path(directory) if directory else config_dir() / "profiles"
    src = directory / f"{key}.json"
    if not src.exists():
        return None
    try:
        return Profile.from_dict(json.loads(src.read_text()))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None                    # a corrupt profile is a missing one
