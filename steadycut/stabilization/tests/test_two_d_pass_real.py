"""The 2D pass on REAL footage: the delivered clip must be the better of the two.

The existing 2D test runs on a synthetic clip with a slow ramp plus a clean 5 Hz
vertical jitter, and the pass removes it, because that is the motion the pass was
tuned against. Singletrack is not that motion: parallax between canopy and
trail, a rider in the near field, foliage moving for reasons unrelated to the
camera. So the synthetic test passes while telling you nothing about the trail.

Measured from the middle of three real rides, at three fractions of each, nine
windows: the pass made the far-field residual WORSE on nine of nine, by 3 to 51
percent. And the old behaviour warned about it and then shipped the worse file
anyway, so the default output of image-only mode was the degraded one.

The invariant worth testing is therefore not "the pass helps" -- it does not, and
a test asserting that would simply be red forever. It is that the clip the caller
receives is the better of raw and corrected, whichever that turns out to be. That
holds whether the pass is fixed, gated off, or eventually re-tuned.

Windows are taken from the MIDDLE of each ride, because the opening seconds are
slow roll-out: the calmest stretch of the 19 May ride is 6.8s in, mean |gyro|
7.15 against a window median of 20.10, and a stabiliser judged on a roll-out is
judged on the least informative material available.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from steadycut.core.pipeline import far_field_motion, stabilize_image_only
from steadycut.ingest.footage import footage_root

# The May rides, at 50% of each file. Two rides keeps the render cost sane; the
# pass's behaviour is not ride-specific (it lost on all three measured).
RIDES = [
    "VID_20260519_074911_00_010_011-Original/VID_20260519_074911_00_010.insv",
    "VID_20260526_093530_00_010_012-Original/VID_20260526_093530_00_010.insv",
]
DUR = 3.0
FRACTION = 0.5


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True,
        check=True).stdout.strip()
    return float(out)


def _rides_present() -> list[Path]:
    return [p for p in (Path(f"{footage_root()}/{n}") for n in RIDES) if p.exists()]


needs_footage = pytest.mark.skipif(not _rides_present(),
                                   reason="source footage not present")


@pytest.mark.parametrize("clip", _rides_present(), ids=lambda p: p.stem[:13])
@needs_footage
def test_the_delivered_clip_is_the_better_of_raw_and_corrected(clip, tmp_path):
    """Whatever the pass did, the file the caller gets is the better one.

    This is the check that was missing. The old code warned when the pass lost
    and shipped the degraded clip, so a caller watching the certificate saw a
    warning and a bad file, and the suite was green throughout.
    """
    start = _duration(clip) * FRACTION
    res = stabilize_image_only(str(clip), tmp_path / "out.mp4", start, DUR)

    raw, fixed = res["raw_bounce"], res["corrected_bounce"]
    kept = res.get("kept")
    expected = "raw" if fixed >= raw else "corrected"
    assert kept == expected, (
        f"{clip.stem} at {start:.0f}s: raw {raw:.2f}, corrected {fixed:.2f}, "
        f"but the delivered clip is {kept!r}")

    # end to end: measure the file that was actually written
    delivered = float(abs(far_field_motion(str(tmp_path / "out.mp4"))).mean())
    best = min(raw, fixed)
    assert delivered <= best * 1.35 + 0.5, (
        f"{clip.stem}: the delivered clip measures {delivered:.2f}, worse than "
        f"the better of raw ({raw:.2f}) and corrected ({fixed:.2f})")
