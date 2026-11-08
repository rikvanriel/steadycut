"""Segmented full-timeline proxy for the clip-picker GUI.

Qt-free: no PySide6 import anywhere in here, so the GUI backend stays
importable from the base install (the blocked-Qt subprocess test covers
this module too). Qt wiring (threads, signals, widgets) lives in app.py;
this module exposes a store, a background builder, a thumb strip, and a
segment player behind plain callbacks and pollable queues.

Layout (30 s segments, 720p everywhere):

- Segment i covers source frames [i*N, (i+1)*N), where N is a frame-count
  constant rounded from the SEGMENT_S window on the source's RATIONAL fps
  (never the proxy cap's rounded 29.97, never seconds: seconds-based
  segments start at a different sub-frame phase every time and the -t cut
  end vs the next -ss start is a coin flip).
- One directory per source identity (<cache>/<fingerprint>/): segment
  files seg_<i>.mp4 plus a per-segment sidecar seg_<i>.json. The directory
  listing IS the index -- no shared mutable manifest for two GUI windows
  to race on. The sidecar (span, pitch, size, fps) is written LAST, after
  the mp4 is atomically published (temp in the same dir + os.replace).
- One frozen pitch per source: resolved once at first open (stored
  policy pitch, else gravity at a fixed reference window), persisted in
  the per-source dir, and stamped into every sidecar. A different pitch
  is a different segment set.
- Segments are video-only (AAC priming adds ~21 ms of leading offset per
  segment) and bounded by -frames:v, never -t.
- Cache policy is distance-from-playhead (named as such, not "LRU"): fill
  NEVER evicts, only demand (playhead / in-out segment) does. Pins cover
  the in/out window, the playhead segment, the preloaded next segment,
  and any segment with an open cap.
- LRV-backed builds: a camera-original beside its pre-stitched single-stream
  sibling renders segments and thumbs from the sibling (dual_stream=False,
  same travel_yaw), ~6x/~13x faster measured. The .insv owns the grid;
  eligibility is fps-rational equality, and built_from rides the sidecar
  only, never the directory identity.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

SEGMENT_S = 30.0
SEGMENT_SIZE = (960, 720)
THUMB_SIZE = (480, 270)
THUMB_EVERY_S = 10.0
SETTLE_S = 0.30
BUILD_NICE = 10
SIDECAR_VERSION = 1
PITCH_FILE = "pitch.json"
THUMB_SHEET = "thumbs.jpg"
THUMB_INDEX = "thumbs.json"
TMP_PREFIX = "tmp_seg_"
TMP_THUMB_PREFIX = "tmp_thumb_"
# A 30 min file at proxy bitrate is ~6-7 GB (30 s 720p ultrafast is
# ~110 MB on real footage), so the default cap fits a whole 30 min
# source with headroom. Fill never grows the dir past this; demand
# evicts farthest-first past it.
DEFAULT_CAP_BYTES = 8 * 1024 ** 3
# Size guess before any segment exists (a 30 s 720p ultrafast segment is
# ~110 MB on real footage); replaced by the measured mean after the
# first build. Only gates fill, never demand.
EST_BYTES_PER_SEGMENT = 128 * 1024 ** 2

_FPS_CACHE: dict[str, Fraction] = {}
_NFRAMES_CACHE: dict[str, int] = {}

_FFPROBE_TIMEOUT = 30.0


def fps_ratio(source: str | Path,
              timeout: float = _FFPROBE_TIMEOUT) -> Fraction:
    """Source frame rate as an exact rational, cached.

    Probed (r_frame_rate, avg_frame_rate fallback), never assumed:
    30000/1001 is not 29.97, and the frame grid the segments live on is
    defined by the exact ratio. Raises ValueError when ffprobe reports
    nothing usable.
    """
    key = str(source)
    if key not in _FPS_CACHE:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                 "-of", "default=noprint_wrappers=1", str(source)],
                capture_output=True, text=True, check=True,
                timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            raise ValueError(
                f"cannot probe {source}: ffprobe timed out after {timeout:g}s"
            ) from None
        ratio = Fraction(0, 1)
        for line in out.splitlines():
            name, _, value = line.partition("=")
            if name.strip() not in ("r_frame_rate", "avg_frame_rate"):
                continue
            try:
                num, _, den = value.strip().partition("/")
                ratio = Fraction(int(num), int(den))
            except (ValueError, ZeroDivisionError):
                continue
            if ratio > 0:
                break
        if ratio <= 0:
            raise ValueError(f"no usable frame rate for {source}")
        _FPS_CACHE[key] = ratio
    return _FPS_CACHE[key]


def total_source_frames(source: str | Path, fps: Fraction,
                        timeout: float = _FFPROBE_TIMEOUT) -> int:
    """Source length in frames: nb_frames when the container has it.

    Falls back to round(duration * fps) for containers without a frame
    count; cached per source. The last segment is clipped to this, so an
    over-estimate only costs a short final build, never a gap.
    """
    key = str(source)
    if key not in _NFRAMES_CACHE:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames,duration",
                 "-of", "default=noprint_wrappers=1", str(source)],
                capture_output=True, text=True, check=True,
                timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            raise ValueError(
                f"cannot probe {source}: ffprobe timed out after {timeout:g}s"
            ) from None
        fields = dict(line.partition("=")[::2]
                      for line in out.splitlines() if "=" in line)
        try:
            total = int((fields.get("nb_frames") or "0").strip())
        except ValueError:
            total = 0
        if total <= 0:
            from steadycut.job import media_duration

            total = max(1, round(media_duration(str(source)) * fps))
        _NFRAMES_CACHE[key] = total
    return _NFRAMES_CACHE[key]


# Sibling LRV: <stem>_00_<chap>.insv pairs with LRV<rest>_01_<chap>.lrv in
# the same directory (the chapter number is kept, never just any LRV in
# the dir). The VID_/LRV_ prefixes are the camera's own naming; matching
# is on structure, and no recording name appears here (hygiene gate).
_LRV_RE = re.compile(r"^VID_(.+)_00_(\d+)\.insv$")


def resolve_lrv(source: str | Path) -> Path | None:
    """Sibling pre-stitched LRV for a camera-original, or None.

    Pure path logic plus an existence check: no probing, so a missing
    sibling (short clips) or a non-conforming name resolves to None and
    the .insv path serves unchanged.
    """
    m = _LRV_RE.match(Path(source).name)
    if m is None:
        return None
    sib = Path(source).parent / f"LRV_{m.group(1)}_01_{m.group(2)}.lrv"
    return sib if sib.is_file() else None


def built_from_for(source: str | Path) -> dict:
    """Build-source identity for a segment sidecar: what made the pixels."""
    kind = "lrv" if str(source).lower().endswith(".lrv") else "insv"
    try:
        st = os.stat(source)
        size, mtime = st.st_size, st.st_mtime
    except OSError:
        size, mtime = 0, 0.0
    return {"kind": kind, "size": size, "mtime": mtime}


def resolve_build_source(source: str | Path) -> tuple[str, int]:
    """(LRV path, LRV frame count) preferred for `source`, or ("", 0).

    Eligible means the LRV's fps rational EQUALS the .insv's: sessions
    whose LRV runs at half rate never serve (the resolver still finds
    them). Frame counts are not compared -- the .insv count owns the
    grid, a one-longer LRV is tolerated, and spans past a short LRV's end
    fall back to the .insv per segment. Anything unusable resolves to the
    .insv path.
    """
    lrv = resolve_lrv(source)
    if lrv is None:
        return "", 0
    try:
        fps = fps_ratio(source)
        if fps != fps_ratio(lrv):
            return "", 0
        return str(lrv), total_source_frames(lrv, fps)
    except (ValueError, subprocess.SubprocessError, OSError):
        # A truncated or still-copying sibling LRV fails the ffprobe
        # (CalledProcessError from check=True); an unreadable one fails
        # the stat. Either way the .insv path serves, so a bad LRV never
        # stops a good .insv from opening its proxy.
        return "", 0


def segment_frame_count(seg_s: float = SEGMENT_S, fps: Fraction = Fraction(30000, 1001)) -> int:
    """Frames per segment: ONE constant for the whole source.

    A single frame-count constant, never variable spans around the seek:
    variable spans break segment identity and fragment the cache.
    """
    return max(1, round(seg_s * fps))


def n_segments(total_frames: int, n: int) -> int:
    """How many segments cover `total_frames` source frames."""
    return max(1, math.ceil(total_frames / n))


def segment_span(i: int, n: int, total_frames: int) -> tuple[int, int]:
    """Source frames [start, end) for segment i, clipped to the file."""
    start = i * n
    if start >= total_frames:
        raise IndexError(f"segment {i} past {total_frames} frames")
    return start, min(start + n, total_frames)


def segment_of_frame(frame: int, n: int) -> int:
    """Which segment owns source frame `frame`."""
    return max(0, frame // n)


def segment_at_time(t: float, fps: Fraction, n: int) -> int:
    """Which segment owns media time `t` (global, seconds)."""
    return max(0, segment_of_frame(int(t * fps), n))


def frame_time(frame: int, fps: Fraction) -> float:
    """Media time of a source frame, full precision (repr round-trips)."""
    return float(Fraction(frame, 1) / fps)


def segment_window(i: int, n: int, fps: Fraction,
                   total_frames: int) -> tuple[float, float, int]:
    """(start_s, dur_s, count) for segment i: exact frame-grid times."""
    start_frame, end_frame = segment_span(i, n, total_frames)
    count = end_frame - start_frame
    return frame_time(start_frame, fps), float(Fraction(count, 1) / fps), count


def seg_names(i: int) -> tuple[str, str]:
    """(video, sidecar) file names for segment i."""
    return f"seg_{i:04d}.mp4", f"seg_{i:04d}.json"


def expected_sidecar(i: int, n: int, fps: Fraction, pitch: float,
                     size: tuple[int, int], total_frames: int,
                     built_from: dict | None = None) -> dict:
    """The sidecar a valid segment i must carry (the segment identity)."""
    start_frame, end_frame = segment_span(i, n, total_frames)
    sidecar = {"version": SIDECAR_VERSION,
               "span_frames": [start_frame, end_frame],
               "frames": end_frame - start_frame,
               "pitch": float(pitch),
               "size": [int(size[0]), int(size[1])],
               "fps": f"{fps.numerator}/{fps.denominator}"}
    if built_from is not None:
        sidecar["built_from"] = dict(built_from)
    return sidecar


def sidecar_matches(sidecar: dict, want: dict) -> bool:
    """True when a segment file on disk is this segment set's segment i.

    Pitch, size, and fps are IN the key: anything that changes pixels
    invalidates the file instead of mixing two segment sets on one
    timeline. The version pins the schema; `frames` is informational
    (the last segment is clipped, so the count alone never identifies).
    `built_from` is deliberately NOT in the key: an LRV appearing or
    disappearing must not orphan existing segments (freshness is checked
    separately, per sidecar, in sidecar_source_fresh).
    """
    if not isinstance(sidecar, dict):
        return False
    if sidecar.get("version") != want.get("version"):
        return False
    for key in ("span_frames", "pitch", "size", "fps"):
        if sidecar.get(key) != want.get(key):
            return False
    return True


def sidecar_source_fresh(sidecar: dict,
                         lrv_stat: tuple[int, float] | None) -> bool:
    """True when a segment's recorded build source still stands.

    An .insv-built segment is accepted always, and so is an old sidecar
    without the field (it reads as kind=insv, so existing caches survive
    a sibling LRV appearing). An LRV-built segment holds only while the
    LRV's size+mtime still match the recording: a replaced or removed
    LRV rebuilds from the .insv instead of serving stale pixels.
    """
    built = sidecar.get("built_from") or {}
    if built.get("kind") != "lrv":
        return True
    if lrv_stat is None:
        return False
    size, mtime = lrv_stat
    return built.get("size") == size and float(built.get("mtime", -1)) == mtime


@dataclass
class SegmentStore:
    """One per-source cache dir: frame-grid planner + demand/evict policy.

    `root` is <cache>/<manifest fingerprint>/; `manifest` is the frozen
    proxy manifest (source identity, camera, stitch, renderer) that names
    the dir. Freshness is per-segment sidecars, not the manifest.
    `source` is the canonical file (the .insv: it owns the frame grid);
    `lrv`/`lrv_frames` name the preferred pre-stitched build source when
    a sibling LRV is eligible ("" / 0: the .insv serves every segment).
    """

    root: Path
    fps: Fraction
    frames_per_segment: int
    total_frames: int
    pitch: float
    size: tuple[int, int] = SEGMENT_SIZE
    cap_bytes: int = DEFAULT_CAP_BYTES
    source: str = ""
    lrv: str = ""
    lrv_frames: int = 0

    def seg_paths(self, i: int) -> tuple[Path, Path]:
        """(video, sidecar) paths for segment i."""
        video, sidecar = seg_names(i)
        return self.root / video, self.root / sidecar

    def count(self) -> int:
        """How many segments cover this source."""
        return n_segments(self.total_frames, self.frames_per_segment)

    def source_for_segment(self, i: int) -> str:
        """Build source for segment i: the LRV when it covers the span.

        The .insv count owns the grid, so a short LRV only serves the
        segments inside its own frame count; spans past its end fall back
        to the .insv. A one-longer LRV covers every span (tolerated).
        An LRV that vanished mid-session falls back too: the plan is
        frozen at open but the file is live, so a missing LRV serves the
        .insv until reopen instead of handing the rebuild a dead path.
        """
        if self.lrv and self.lrv_frames \
                and os.path.isfile(self.lrv):
            _, end = segment_span(i, self.frames_per_segment,
                                  self.total_frames)
            if end <= self.lrv_frames:
                return self.lrv
        return self.source

    def expected_for(self, i: int, src: str | Path) -> dict:
        """The sidecar segment i carries when built from `src`."""
        return expected_sidecar(i, self.frames_per_segment, self.fps,
                                self.pitch, self.size, self.total_frames,
                                built_from=built_from_for(src))

    def expected(self, i: int) -> dict:
        """The sidecar segment i must carry to count as present."""
        return self.expected_for(i, self.source_for_segment(i))

    def _lrv_stat(self) -> tuple[int, float] | None:
        """(size, mtime) of the LRV we'd build from now, or None.

        The candidate is the sibling resolved at open (a stat, not a
        probe); the LRV set is fixed for the store's lifetime, so an LRV
        appearing mid-session takes effect on reopen, never mid-stream.
        An LRV vanishing mid-session is the other direction: the
        per-segment/per-time source pick falls back to the .insv (the
        stat here goes None, so LRV-built sidecars read stale), and no
        rebuild is ever handed the missing path. A directly-opened .lrv
        is its own candidate.
        """
        cand = self.lrv
        if not cand and str(self.source).lower().endswith(".lrv"):
            cand = str(self.source)
        if not cand:
            return None
        try:
            st = os.stat(cand)
        except OSError:
            return None
        return (st.st_size, st.st_mtime)

    def _fresh(self, cached: dict) -> bool:
        """True when a sidecar's recorded build source still stands."""
        return sidecar_source_fresh(cached, self._lrv_stat())

    def is_present(self, i: int) -> bool:
        """True when segment i is on disk AND matches this segment set."""
        try:
            video, sidecar = self.seg_paths(i)
        except IndexError:
            return False
        if not video.exists() or not sidecar.exists():
            return False
        try:
            cached = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            return False
        return sidecar_matches(cached, self.expected(i)) \
            and self._fresh(cached)

    def scan(self) -> dict[int, dict]:
        """Index from the directory listing: {i: sidecar} for valid files.

        No shared mutable "what exists" manifest (two GUI windows on one
        source would race it); the listing plus per-segment sidecars IS
        the index. A sidecar written by a crashed window never validates,
        because the sidecar lands last, after the rename.
        """
        found: dict[int, dict] = {}
        if not self.root.is_dir():
            return found
        for sidecar_path in sorted(self.root.glob("seg_*.json")):
            try:
                i = int(sidecar_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            try:
                cached = json.loads(sidecar_path.read_text())
            except (OSError, ValueError):
                continue
            if sidecar_matches(cached, self.expected(i)) \
                    and self._fresh(cached) \
                    and self.seg_paths(i)[0].exists():
                found[i] = cached
        return found

    def clean_temps(self) -> int:
        """Remove staged temp files (crashed windows never rename)."""
        removed = 0
        if not self.root.is_dir():
            return removed
        for tmp in list(self.root.glob(f"{TMP_PREFIX}*")) \
                + list(self.root.glob(f"{TMP_THUMB_PREFIX}*")):
            try:
                tmp.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def dir_size(self) -> int:
        """Bytes under the per-source dir (segments + thumbs)."""
        total = 0
        if not self.root.is_dir():
            return total
        for path in self.root.iterdir():
            if path.is_file() and not path.name.startswith(TMP_PREFIX) \
                    and not path.name.startswith(TMP_THUMB_PREFIX):
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    def est_bytes(self) -> int:
        """Per-segment size guess: measured mean, else the constant."""
        present = self.scan()
        sizes = []
        for i in present:
            try:
                sizes.append(self.seg_paths(i)[0].stat().st_size)
            except OSError:
                pass
        if sizes:
            return max(1, sum(sizes) // len(sizes))
        return EST_BYTES_PER_SEGMENT

    def pins(self, playhead_i: int, in_i: int | None = None,
             out_i: int | None = None,
             open_segments: tuple[int, ...] = ()) -> set[int]:
        """Unevictable segments: in/out window, playhead, preload, caps.

        Otherwise in/out brackets an evicted region and loop playback hits
        "building" every pass.
        """
        keep = {playhead_i, playhead_i + 1, *open_segments}
        if in_i is not None and out_i is not None:
            keep.update(range(min(in_i, out_i), max(in_i, out_i) + 1))
        return {i for i in keep if 0 <= i < self.count()}

    def enforce_cap(self, keep: set[int],
                    rank: list[int] | None = None) -> list[int]:
        """Evict until under the cap; returns evicted indices, farthest first.

        Only DEMAND (playhead / in-out segment) calls this: fill NEVER
        evicts (if the file at proxy bitrate exceeds the cap, fill-into-
        evict loops forever). `rank` orders victims farthest-first and
        defaults to index order; pinned segments are never touched even
        when that leaves the dir over cap (a cap smaller than the window
        is a configuration error, not an eviction).
        """
        if self.dir_size() <= self.cap_bytes:
            return []
        victims = rank if rank is not None else sorted(self.scan())
        evicted = []
        for i in victims:
            if i in keep:
                continue
            video, sidecar = self.seg_paths(i)
            if not video.exists():
                continue
            try:
                video.unlink()
                sidecar.unlink(missing_ok=True)
                evicted.append(i)
            except OSError:
                continue
            if self.dir_size() <= self.cap_bytes:
                break
        return evicted

    def adopt_or_delete_legacy(self, legacy_mp4: str | Path,
                               timeout: float = _FFPROBE_TIMEOUT) -> str:
        """Adopt the old single-file proxy as segment 0, or delete it.

        The old <fp>.mp4 layout is orphaned by <fp>/seg_<i>.mp4. It only
        counts as segment 0 when it matches the frame-grid definition
        (exact frame count, video-only, same span): with one constant
        frame count per source that is essentially never, so this is
        normally a delete on first open.
        Returns "adopted", "deleted", or "absent".
        """
        legacy = Path(legacy_mp4)
        if not legacy.exists():
            return "absent"
        adopted = False
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames,width,height,duration",
                 "-show_entries", "stream=codec_type",
                 "-of", "json", str(legacy)],
                capture_output=True, text=True, check=True,
                timeout=timeout).stdout
            streams = json.loads(out).get("streams", [])
            video = [s for s in streams if s.get("codec_type") == "video"]
            audio = [s for s in streams if s.get("codec_type") == "audio"]
            if len(video) == 1 and not audio:
                have = int(video[0].get("nb_frames") or 0)
                want = self.expected(0)["frames"]
                dur = float(video[0].get("duration") or 0.0)
                want_dur = float(Fraction(want, 1) / self.fps)
                size_ok = (int(video[0].get("width") or 0),
                           int(video[0].get("height") or 0)) == tuple(self.size)
                if have == want and abs(dur - want_dur) * float(self.fps) < 0.5 \
                        and size_ok and self.expected(0)["span_frames"][0] == 0:
                    dst, sidecar = self.seg_paths(0)
                    self.root.mkdir(parents=True, exist_ok=True)
                    os.replace(legacy, dst)
                    # The legacy file is an .insv render: label it as built
                    # from the canonical source, whatever the planner would
                    # prefer today (an LRV appearing must not orphan it).
                    sidecar.write_text(json.dumps(
                        self.expected_for(0, self.source), indent=1))
                    adopted = True
        except (OSError, ValueError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            adopted = False
        if adopted:
            return "adopted"
        try:
            legacy.unlink()
        except OSError:
            pass
        return "deleted"


def load_or_resolve_pitch(store_dir: str | Path, source: str | Path,
                          profile=None, reference_start: float = 0.0,
                          gravity=None) -> tuple[float, dict]:
    """ONE pitch per source: persisted, reused, stamped everywhere.

    The stored value wins across sessions (a stored pitch can change
    between sessions while pitch is not in the fingerprint, so re-
    measuring at open would mix segments built at two pitches). First
    open resolves the stored policy pitch, else gravity at a fixed
    reference window, else the constant fallback -- and persists it.
    `gravity` injects (pitch, info) measurement for tests.
    """
    pitch_path = Path(store_dir) / PITCH_FILE
    try:
        cached = json.loads(pitch_path.read_text())
        return float(cached["pitch"]), {"layer": "cache",
                                       "usable": True}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if gravity is not None:
        pitch, info = gravity()
    else:
        from steadycut.gui import backend as B

        pitch, info = B.frozen_pitch(str(source), reference_start, profile)
    try:
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        pitch_path.write_text(json.dumps({"pitch": float(pitch),
                                         "info": dict(info)}, indent=1))
    except OSError:
        pass
    return pitch, info


def fill_order(playhead_i: int, total: int,
               present: set[int]) -> list[int]:
    """Missing segments playhead-outward (nearest first, ties to the right)."""
    missing = [i for i in range(total) if i not in present]
    return sorted(missing, key=lambda i: (abs(i - playhead_i), i < playhead_i))


@dataclass
class _Demand:
    playhead_i: int | None = None
    in_i: int | None = None
    out_i: int | None = None
    seq: int = 0


class SegmentBuilder:
    """Background segment builds: debounced demand, playhead-outward fill.

    Demand (seek / in-out window) jumps the queue after the playhead
    settles (~300 ms debounce); in-flight builds are never cancelled for
    a new playhead (a bounded segment costs less than a churned one).
    Fill runs playhead-outward and never evicts. During renders no new
    builds start (in-flight finishes). Cancel is group-kill, same as
    Render. Events land on a pollable queue: ("segment", i, ok, detail).
    """

    def __init__(self, store: SegmentStore, build_fn=None,
                 settle_s: float = SETTLE_S):
        self.store = store
        self.settle_s = settle_s
        self.build_fn = build_fn or default_build_fn
        self.events: queue.Queue = queue.Queue()
        self._demand = _Demand()
        self._demand_at = 0.0
        self._rendering = False
        self._failed: set[int] = set()
        self._stop = threading.Event()
        self._wake = threading.Condition()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._open: tuple[int, ...] = ()

    def start(self) -> None:
        """Launch the worker thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="proxy-fill")
        self._thread.start()

    def stop(self, cancel_in_flight: bool = True) -> None:
        """Stop: drop pending work, group-kill the in-flight build.

        Group-kill (not a cooperative flag) because a segment build is
        minutes of ffmpeg: closing the window must not wait for it.
        """
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        if cancel_in_flight:
            self._kill_in_flight()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None

    def note_open_segments(self, open: tuple[int, ...]) -> None:
        """Segments the player currently holds caps on: unevictable.

        The player keeps the current segment plus preload open across
        boundary crossings; evicting under an open cap turns a cached
        seek into a "building" miss. The app refreshes this from
        player.open_segments() on its poll loop (best-effort: a stale
        set only over-pins until the next poll).
        """
        with self._wake:
            self._open = tuple(open)

    def request_playhead(self, t: float) -> None:
        """Demand the segment under the playhead (debounced)."""
        i = segment_at_time(t, self.store.fps, self.store.frames_per_segment)
        with self._wake:
            self._demand.playhead_i = i
            self._demand.seq += 1
            self._demand_at = time.monotonic()
            self._failed.discard(i)
            self._wake.notify_all()

    def request_window(self, in_t: float, out_t: float) -> None:
        """Demand the in/out window (pins + priority, still debounced)."""
        n = self.store.frames_per_segment
        in_i = segment_at_time(min(in_t, out_t), self.store.fps, n)
        out_i = segment_at_time(max(in_t, out_t), self.store.fps, n)
        with self._wake:
            self._demand.in_i, self._demand.out_i = in_i, out_i
            self._demand.seq += 1
            self._demand_at = time.monotonic()
            self._failed.difference_update(range(in_i, out_i + 1))
            self._wake.notify_all()

    def set_rendering(self, active: bool) -> None:
        """Pause starting new builds during renders (in-flight finishes)."""
        with self._wake:
            self._rendering = active
            self._wake.notify_all()

    def _kill_in_flight(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            from steadycut.gui import runner as R

            try:
                R.cancel(proc, grace_s=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _targets(self) -> tuple[list[int], set[int]]:
        """(build order, pins): demand first, then playhead-outward fill."""
        store = self.store
        d = self._demand
        playhead_i = d.playhead_i if d.playhead_i is not None else 0
        in_i, out_i = d.in_i, d.out_i
        present = set(store.scan())
        demand: list[int] = []
        if d.playhead_i is not None and d.playhead_i not in present:
            demand.append(d.playhead_i)
        if in_i is not None and out_i is not None:
            demand += [i for i in range(min(in_i, out_i),
                                       max(in_i, out_i) + 1)
                       if i not in present and i not in demand]
        fill = [i for i in fill_order(playhead_i, store.count(), present)
                if i not in demand]
        with self._wake:
            open_segs = self._open
        return ([i for i in demand + fill if i not in self._failed],
                store.pins(playhead_i, in_i, out_i, open_segs))

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._wake:
                self._wake.wait(timeout=0.05)
            if self._stop.is_set():
                return
            with self._wake:
                settled = (time.monotonic() - self._demand_at
                           >= self.settle_s)
                rendering = self._rendering
            targets, pins = self._targets()
            if not targets:
                continue
            if rendering:
                continue  # paused: never start new builds mid-render
            with self._wake:
                demand = self._demand
                demand_missing = (
                    demand.playhead_i is not None
                    and not self.store.is_present(demand.playhead_i))
            if demand_missing and not settled:
                continue  # playhead still moving: build nothing yet
            # Fill into an over-cap dir loops forever (evict-then-rebuild),
            # so fill waits for room while demand evicts after building.
            i = targets[0]
            is_demand = i in self._demand_set()
            if not is_demand and \
                    self.store.dir_size() + self.store.est_bytes() \
                    > self.store.cap_bytes:
                continue
            try:
                self.build_fn(self.store, i, self)
            except Exception as exc:  # failed: park it, keep filling others
                self._failed.add(i)
                self.events.put(("segment", i, False, f"{exc}"))
                continue
            # A reprioritize that landed mid-build wins by order only: the
            # finished segment still publishes (atomic); targets re-derive
            # next pass.
            self.events.put(("segment", i, True, ""))
            if is_demand:
                self.store.enforce_cap(pins,
                                       rank=self._evict_rank(pins))

    def _demand_set(self) -> set[int]:
        d = self._demand
        want = set()
        if d.playhead_i is not None:
            want.add(d.playhead_i)
        if d.in_i is not None and d.out_i is not None:
            want.update(range(min(d.in_i, d.out_i),
                              max(d.in_i, d.out_i) + 1))
        return want

    def _evict_rank(self, pins: set[int]) -> list[int]:
        playhead_i = self._demand.playhead_i or 0
        present = [i for i in self.store.scan() if i not in pins]
        return sorted(present,
                      key=lambda i: (abs(i - playhead_i), i < playhead_i),
                      reverse=True)

    def drain(self) -> list:
        """Pending events (non-blocking)."""
        out = []
        try:
            while True:
                out.append(self.events.get_nowait())
        except queue.Empty:
            pass
        return out


def default_build_fn(store: SegmentStore, i: int,
                      builder: SegmentBuilder | None = None) -> None:
    """Build segment i in an isolated process group (group-killable).

    A fresh interpreter per segment: spawn/seek cost is the `a` in the
    measured wall (a + b*dur), and a crashed ffmpeg can never take the
    GUI thread with it. Runs at nice +10 so fill doesn't stutter the
    UI's own decode; the child owns its process group for cancel.
    The build source is the planner's per-segment pick (eligible LRV
    when it covers the span, else the .insv), and the sidecar records
    which file the pixels came from.
    """
    start_frame, end_frame = segment_span(i, store.frames_per_segment,
                                          store.total_frames)
    count = end_frame - start_frame
    video, sidecar = store.seg_paths(i)
    store.root.mkdir(parents=True, exist_ok=True)
    tmp = video.with_name(f"{TMP_PREFIX}{video.name}")
    cmd = [sys.executable, "-m", "steadycut.gui.proxy_build",
           "--source", str(store.source_for_segment(i)),
           "--frame-start", str(start_frame),
           "--nframes", str(count),
           "--fps", f"{store.fps.numerator}/{store.fps.denominator}",
           "--pitch", repr(float(store.pitch)),
           "--width", str(store.size[0]), "--height", str(store.size[1]),
           "--out", str(video), "--tmp", str(tmp),
           "--sidecar", str(sidecar),
           "--sidecar-json", json.dumps(store.expected(i))]

    def _preexec() -> None:
        os.setsid()
        try:
            os.nice(BUILD_NICE)
        except OSError:
            pass

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, preexec_fn=_preexec)
    if builder is not None:
        builder._proc = proc
    try:
        text = proc.communicate()[0] or b""
    finally:
        if builder is not None and builder._proc is proc:
            builder._proc = None
    if proc.returncode != 0:
        raise RuntimeError(f"segment {i} build exited "
                           f"{proc.returncode}: "
                           f"{text.decode('utf-8', 'replace')[-2000:]}")


def open_store(root: str | Path, source: str | Path, pitch: float,
               size: tuple[int, int] = SEGMENT_SIZE,
               seg_s: float = SEGMENT_S,
               cap_bytes: int = DEFAULT_CAP_BYTES) -> SegmentStore:
    """Open (or adopt) the per-source cache dir: probe once, plan the grid.

    The grid (fps, frame count) always comes from the canonical source;
    an eligible sibling LRV only picks the build pixels, per segment.
    """
    fps = fps_ratio(source)
    n = segment_frame_count(seg_s, fps)
    total = total_source_frames(source, fps)
    lrv, lrv_frames = resolve_build_source(source)
    store = SegmentStore(root=Path(root), fps=fps, frames_per_segment=n,
                         total_frames=total, pitch=float(pitch),
                         size=tuple(size), cap_bytes=cap_bytes,
                         lrv=lrv, lrv_frames=lrv_frames)
    store.source = str(source)
    store.root.mkdir(parents=True, exist_ok=True)
    store.clean_temps()
    return store


class ThumbStrip:
    """Full-timeline thumbnail strip: sparse grabs streamed into a sheet.

    One frame per THUMB_EVERY_S across the whole file, each through the
    real render path at thumb size with the SAME frozen pitch as the
    proxy (raw fisheye stills don't show where a clip is). Grabs run in a
    small worker pool and stream in: the gate is time-to-first-thumbs,
    not the full strip (minutes serial). The contact sheet + index cache
    beside the proxy by manifest fingerprint -- never 180 loose PNGs.
    Events on a pollable queue: ("thumb", t, jpg_bytes), ("done",).
    """

    def __init__(self, source: str | Path, times: list[float], pitch: float,
                 size: tuple[int, int] = THUMB_SIZE,
                 sheet: str | Path | None = None,
                 index: str | Path | None = None,
                 grab_fn=None, workers: int = 4):
        self.source = str(source)
        self.times = list(times)
        self.pitch = float(pitch)
        self.size = tuple(size)
        self.sheet = Path(sheet) if sheet is not None else None
        self.index = Path(index) if index is not None else None
        self.grab_fn = grab_fn or grab_thumb
        self.workers = max(1, workers)
        # Preferred grab pixels: the eligible sibling LRV when one serves
        # this source (resolved once; per-thumb fallback below). No extra
        # probes when no sibling exists: the resolver is path-only first.
        self._lrv, self._lrv_frames = resolve_build_source(source)
        self.events: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._dirty = False
        self.cells: dict[float, bytes] = {}
        self._load_sheet()

    def _load_sheet(self) -> None:
        """Instant thumbs on reopen: decode the cached contact sheet."""
        if self.sheet is None or self.index is None:
            return
        try:
            meta = json.loads(self.index.read_text())
        except (OSError, ValueError):
            return
        if meta.get("pitch") != float(self.pitch) or \
                meta.get("size") != [self.size[0], self.size[1]]:
            return  # different framing: rebuild, never mix
        try:
            import cv2
            import numpy as np

            sheet = cv2.imread(str(self.sheet), cv2.IMREAD_COLOR)
        except ImportError:
            return
        if sheet is None:
            return
        cols = int(meta.get("cols") or 0)
        cell = meta.get("cell") or [0, 0]
        slots = meta.get("times") or []
        if not cols or not cell[0]:
            return
        for k, t in enumerate(slots):
            if t not in self.times:
                continue
            r, c = divmod(k, cols)
            y, x = r * cell[1], c * cell[0]
            crop = sheet[y:y + cell[1], x:x + cell[0]]
            if crop.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                with self._lock:
                    self.cells[float(t)] = bytes(buf)

    def source_for_time(self, t: float) -> str:
        """Grab source for a thumb at `t`: the LRV while it covers `t`.

        Past the LRV's end the .insv serves (same per-segment fallback as
        the segment planner, at thumb granularity). An LRV that vanished
        mid-session falls back the same way until reopen.
        """
        if self._lrv and self._lrv_frames \
                and os.path.isfile(self._lrv):
            try:
                fps = fps_ratio(self.source)
            except ValueError:
                return self.source
            if t < self._lrv_frames / float(fps):
                return self._lrv
        return self.source

    def start(self) -> None:
        """Grab missing thumbs, streaming; cached cells emit immediately."""
        with self._lock:
            missing = [t for t in self.times if t not in self.cells]
            for t in self.times:
                if t in self.cells:
                    self.events.put(("thumb", t, self.cells[t]))
        self._stop.clear()
        work = queue.Queue()
        for t in missing:
            work.put(t)
        for _ in range(min(self.workers, len(missing) or 1)):
            th = threading.Thread(target=self._run, args=(work,),
                                  daemon=True, name="proxy-thumb")
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        """No new grabs; in-flight ones finish (bounded, ~1 s each)."""
        self._stop.set()
        for th in self._threads:
            th.join(timeout=30.0)
        self._threads = []
        if self._dirty:
            # Flush the tail the throttled periodic saves didn't reach,
            # so reopen reuses the whole strip instead of rebuilding it.
            self.save_sheet()

    def _run(self, work: queue.Queue) -> None:
        while not self._stop.is_set():
            try:
                t = work.get_nowait()
            except queue.Empty:
                return
            try:
                jpg = self.grab_fn(self.source_for_time(t), t, self.pitch,
                                   self.size)
            except Exception as exc:
                self.events.put(("thumb-error", t, f"{exc}"))
                continue
            with self._lock:
                self.cells[float(t)] = jpg
                self._dirty = True
            self.events.put(("thumb", t, jpg))

    def save_sheet(self, cols: int = 18) -> None:
        """Write the contact sheet + index (throttled by the caller)."""
        if self.sheet is None or self.index is None:
            return
        try:
            import cv2
            import numpy as np
        except ImportError:
            return
        with self._lock:
            items = sorted(self.cells.items())
        if not items:
            return
        thumbs = []
        for _, jpg in items:
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            thumbs.append(img)
        if not thumbs:
            return
        h, w = thumbs[0].shape[:2]
        rows = math.ceil(len(thumbs) / cols)
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, img in enumerate(thumbs):
            r, c = divmod(k, cols)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        tmp = self.sheet.with_name(f"{TMP_THUMB_PREFIX}{self.sheet.name}")
        cv2.imwrite(str(tmp), canvas)
        os.replace(tmp, self.sheet)
        meta = {"pitch": float(self.pitch),
                "size": [self.size[0], self.size[1]],
                "cols": cols, "cell": [w, h],
                "times": [t for t, _ in items]}
        tmp_meta = self.index.with_name(
            f"{TMP_THUMB_PREFIX}{self.index.name}")
        tmp_meta.write_text(json.dumps(meta))
        os.replace(tmp_meta, self.index)  # index last: sheet without index is ignored
        self._dirty = False

    def drain(self) -> list:
        """Pending events (non-blocking)."""
        out = []
        try:
            while True:
                out.append(self.events.get_nowait())
        except queue.Empty:
            pass
        return out


def thumb_times(total_s: float, every_s: float = THUMB_EVERY_S) -> list[float]:
    """Sparse grab times across the whole file (centers of each slot)."""
    n = max(1, int(total_s / every_s))
    return [(k + 0.5) * total_s / n for k in range(n)]


def grab_thumb(source: str, t: float, pitch: float,
               size: tuple[int, int] = THUMB_SIZE) -> bytes:
    """One thumbnail through the real render path (frozen pitch).

    Single-frame window at thumb size: the same framing the proxy
    shows, so the strip matches the preview. Returns JPEG bytes.
    """
    import cv2

    from steadycut.job import build_proxy

    fps = fps_ratio(source)
    dur = float(Fraction(1, 1) / fps)
    with tempfile.TemporaryDirectory(prefix="steadycut-thumb-") as tmp:
        dst = str(Path(tmp) / "thumb.mp4")
        build_proxy(source, t, dur, dst, pitch, size=tuple(size),
                    nframes=1, video_only=True)
        cap = cv2.VideoCapture(dst)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"thumb grab at {t:.2f}s produced no frame")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError(f"thumb JPEG encode failed at {t:.2f}s")
    return bytes(buf)


class SegmentPlayer:
    """Playback across segments: one cap per segment, preload, exact swap.

    Both sides share the frozen pitch AND the frame grid, so the handoff
    is exact. Seeks use POS_FRAMES (never POS_MSEC off an unsnapped p0).
    Unbuilt regions hold the last frame and report "building" -- the UI
    never blocks. At most a few caps stay open (current + preload); the
    open ones are pinned in the cache.
    """

    KEEP_CAPS = 3

    def __init__(self, store: SegmentStore):
        import cv2

        self._cv2 = cv2
        self.store = store
        self.caps: dict[int, object] = {}
        self.current: int | None = None
        self.last_frame = None
        self.last_t: float | None = None

    def open_segments(self) -> tuple[int, ...]:
        """Segments with an open cap (cache pins)."""
        return tuple(sorted(self.caps))

    def _open(self, i: int):
        cap = self._cv2.VideoCapture(str(self.store.seg_paths(i)[0]))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open segment {i}")
        self.caps[i] = cap
        while len(self.caps) > self.KEEP_CAPS:
            drop = next(k for k in self.caps if k != self.current and k != i)
            try:
                self.caps.pop(drop).release()
            except Exception:
                pass
        # Preload the neighbour: open it now so the boundary swap is a
        # positioned cap, not a cold open mid-playback.
        nxt = i + 1
        if nxt < self.store.count() and nxt not in self.caps \
                and self.store.is_present(nxt) \
                and len(self.caps) < self.KEEP_CAPS + 1:
            try:
                pre = self._cv2.VideoCapture(str(self.store.seg_paths(nxt)[0]))
                if pre.isOpened():
                    self.caps[nxt] = pre
            except Exception:
                pass
        return cap

    def frame_at(self, t: float) -> tuple[object | None, str]:
        """Frame for media time t: (bgr, status).

        Status is "ok", "building" (unbuilt gap: frame is the held last
        frame, possibly None before the first build), or "eof".
        """
        store = self.store
        total_end = frame_time(store.total_frames, store.fps)
        if t >= total_end:
            return self.last_frame, "eof"
        i = segment_at_time(t, store.fps, store.frames_per_segment)
        if i >= store.count() or not store.is_present(i):
            return self.last_frame, "building"
        start_frame, _ = segment_span(i, store.frames_per_segment,
                                      store.total_frames)
        local = int(t * store.fps) - start_frame
        cap = self.caps.get(i)
        if cap is None:
            try:
                cap = self._open(i)
            except RuntimeError:
                return self.last_frame, "building"
        self.current = i
        cap.set(self._cv2.CAP_PROP_POS_FRAMES, max(0, local))
        ok, frame = cap.read()
        if not ok or frame is None:
            return self.last_frame, "building"
        self.last_frame = frame
        self.last_t = t
        return frame, "ok"

    def close(self) -> None:
        """Release every cap (pins drop with them)."""
        for cap in self.caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self.caps = {}
        self.current = None


@dataclass
class PlaybackCursor:
    """Where the playhead is and what the viewer should show.

    Loop playback wraps at out_t (never at p0): an unbuilt gap inside
    [in, out] holds the last frame instead of wrapping past it.
    """

    play_t: float = 0.0
    in_t: float = 0.0
    out_t: float = 0.0

    def advance(self, dt: float, status: str) -> tuple[float, str]:
        """Next playhead + overlay after a tick that read `status`.

        On "building" the playhead holds (the last frame stays up); on
        reaching out_t it wraps to in_t (in-window loop).
        """
        if status == "building":
            return self.play_t, "building …"
        target = self.play_t + dt
        if target >= self.out_t:
            return self.in_t, ""
        self.play_t = target
        return target, ""
