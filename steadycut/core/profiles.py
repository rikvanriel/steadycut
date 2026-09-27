"""Per-recording framing profile: measurements of a clip, not preferences.

Both of the things stored here are properties of ONE recording. The pitch is a
per-recording constant because the bar's elevation is posture-dependent -- 44.5
degrees below the horizon at one moment, 38.6 at another, on the same ride -- so
a number that holds one recording puts the bar out of frame on the next. The cue
is fitted from a handful of labelled frames of THIS recording, so it learns this
camera's mount and this ride's light. Neither belongs in the source code, and
neither is a user preference.

What is deliberately NOT here: the composition target. Where the boundary should
sit in frame is a CHOICE, and a preference stored among measurements is how a
preference ends up looking like evidence. It defaults in
steadycut.framing.composition and an activity overrides it there.

What the first version lacked and this one has: what the fit was worth (frames
labelled, held-out error) and the source's size and mtime. Without the second
pair a re-export silently reuses a profile fitted on different footage; without
the first there is no way to tell a good profile from a bad one. A missing
profile is not an error -- it means nobody has measured this clip yet, and the
policy falls back to the pitch search it already has.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VERSION = 2


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
    cue: Any = None                     # a fitted GroundCue, or None
    fitted_on: dict = field(default_factory=dict)   # frames, held_out_error
    source_size: int | None = None      # so a re-export cannot reuse silently
    source_mtime: float | None = None
    notes: dict = field(default_factory=dict)

    def record_source(self, source) -> "Profile":
        """Remember which file this was fitted from. Missing file: no claim."""
        try:
            st = Path(source).expanduser().stat()
        except OSError:
            return self
        self.source_size, self.source_mtime = st.st_size, st.st_mtime
        return self

    def matches(self, source) -> bool:
        """True when `source` looks like the file this profile was fitted from.

        No recorded claim means "unchecked", which is not the same as "matches":
        `verified` says which, so a caller can decide how much to trust it.
        """
        if not self.verified:
            return True
        try:
            st = Path(source).expanduser().stat()
        except OSError:
            return False
        return st.st_size == self.source_size and st.st_mtime == self.source_mtime

    @property
    def verified(self) -> bool:
        """Whether a source claim was recorded at all."""
        return self.source_size is not None and self.source_mtime is not None

    def quality(self) -> str:
        """One line a human can judge the profile by."""
        if self.cue is None:
            return "pitch only, no fitted cue"
        frames = self.fitted_on.get("frames", "?")
        err = self.fitted_on.get("held_out_error")
        err_s = "unmeasured" if err is None else f"{err:.3f}"
        return (f"cue fitted on {frames} frames, held-out error {err_s} "
                f"of frame height")

    def to_dict(self) -> dict:
        cue = None
        if self.cue is not None:
            cue = {
                "weights": np.asarray(self.cue.weights).tolist(),
                "mean": np.asarray(self.cue.mean).tolist(),
                "scale": np.asarray(self.cue.scale).tolist(),
            }
        return {"version": VERSION, "pitch": self.pitch, "cue": cue,
                "fitted_on": self.fitted_on, "source_size": self.source_size,
                "source_mtime": self.source_mtime, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        cue = None
        if d.get("cue"):
            from steadycut.framing.ground_cue import GroundCue

            c = d["cue"]
            cue = GroundCue(weights=np.asarray(c["weights"], dtype=float),
                            mean=np.asarray(c["mean"], dtype=float),
                            scale=np.asarray(c["scale"], dtype=float))
        return cls(pitch=d.get("pitch"), cue=cue,
                   fitted_on=d.get("fitted_on", {}),
                   source_size=d.get("source_size"),
                   source_mtime=d.get("source_mtime"),
                   notes=d.get("notes", {}))


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


def load_for(source, directory: Path | None = None) -> Profile | None:
    """The profile for this recording, or None if there is no usable one.

    None covers the three cases a caller should treat alike: nobody has measured
    this clip, the stored profile is corrupt, or the source no longer matches
    what was measured. The last is the re-export case, where framing on the old
    profile would be worse than falling back to the search.
    """
    profile = load(key_for(source), directory=directory)
    if profile is None:
        return None
    return profile if profile.matches(source) else None
