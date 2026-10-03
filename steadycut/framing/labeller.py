"""The labelling pass `GroundCue.fit()` needs, and that nothing implemented.

`fit()` takes `(image, boundary per column)` pairs. Nothing in the tree produces
the boundary half -- the plan called it "the caller's business" and it was never
built -- which is the entire reason the cue has never run on real footage. Its own
tests fit on SYNTHETIC frames with a boundary known by construction, so the
"within 0.082 of frame height" in its docstring is a synthetic number that has
never been earned on this material. The cue is a well-designed model waiting for
labels; this is the thing that supplies them.

WHY SPARSE CLICKS ARE THE RIGHT INPUT, not a compromise. `fit()` skips unlabelled
pixels, so a boundary drawn across the middle third of the frame and NaN
everywhere else is a first-class label. Nobody marks 960 columns by hand, and they
should not: the boundary is a smooth curve, a person marks where it is obviously
wrong, and the fit learns the rest from the pixels. The per-column array is the
INTERFACE, not the interaction.

ONE CLICK IS REFUSED, deliberately. A single point can be interpolated into a
confident-looking flat line, and a fit over a flat line will report success while
learning nothing about where the ground is. That is the single most likely way a
labelling pass produces a confidently wrong cue, so the assembly step returns NaN
and `usable()` rejects it rather than inventing the surface.

BOUNDARIES ARE FRACTIONS OF HEIGHT, matching what `fit()` documents. Pixels would
be a factor-of-H error that still fits, still converges, and still returns a
number that looks reasonable until it is compared against a fraction.

The window is OpenCV's own, not a GUI toolkit: the project already depends on
OpenCV, and `setMouseCallback` needs nothing new. The interactive shell is
deliberately thin -- every behaviour worth testing is a plain function above it,
and the callbacks only collect points and call them.

Run:
    python -m steadycut.framing.labeller VIDEO.mp4 --samples labels.npz
    python -m steadycut.framing.labeller --fit labels.npz --source VIDEO.mp4 \
        --profile ~/.config/steadycut/profiles/<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - environment guard
    # This bites on the system interpreter, which has no OpenCV, and the bare
    # "No module named 'cv2'" gives no hint that the fix is a different python.
    sys.exit(
        "cv2 is not installed for this interpreter.\n"
        "Use the project's virtualenv:\n"
        "    .venv/bin/steadycut-label ...        (preferred)\n"
        f"or  {sys.executable} -m steadycut.framing.labeller ...")

import numpy as np

from steadycut.framing.ground_cue import GroundCue

# Keys the window listens for. Left click adds a point; these finish the frame.
KEY_ACCEPT = 13        # return
KEY_SKIP = ord("s")
KEY_QUIT = 27          # escape
KEY_UNDO = ord("u")


def usable(boundary) -> bool:
    """True when a boundary carries at least one labelled column.

    One click cannot make a boundary, and the check lives here rather than at the
    call site because the failure it prevents -- a fit that converges on a flat
    line and reports success -- is silent.
    """
    b = np.asarray(boundary, dtype=float)
    return b.ndim == 1 and np.isfinite(b).sum() >= 2


def boundary_from_points(points, width: int, height: int) -> np.ndarray:
    """A per-column boundary (fraction of height), NaN outside the clicked span.

    `points` are (x, y) in PIXELS, left to right; consecutive points are joined
    linearly, because the ground boundary is smooth and a person marking six
    points is describing a curve, not six samples of one. Columns before the
    first click and after the last stay NaN: the fit skips them, and guessing at
    them would be inventing labels nobody made.

    Raises on a point outside the frame. A mis-click above the top edge is not a
    boundary at -0.4 of height, and clamping it would manufacture a label.
    """
    b = np.full(int(width), np.nan)
    pts = sorted((int(x), float(y)) for x, y in points)
    if len(pts) < 2:
        return b
    for x, y in pts:
        if not 0 <= x < int(width):
            raise ValueError(f"click x={x} is outside the frame (width {width})")
        if not 0.0 <= y <= float(height):
            raise ValueError(
                f"click y={y} is outside the frame (height {height}); it is a "
                f"mis-click, not a boundary above the top edge")
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float) / float(height)
    lo, hi = int(xs[0]), int(xs[-1])
    b[lo:hi + 1] = np.interp(np.arange(lo, hi + 1), xs, ys)
    return b


def samples_to_npz(samples, path) -> str:
    """Write (image, boundary) pairs, losslessly, to one file.

    Images go in as PNG bytes rather than arrays so the file stays small and the
    round trip is bit-exact: a labelling pass is long enough, and repeated
    labelling on the same clip normal, that losing a frame to a lossy format is a
    cost the caller pays for no benefit.
    """
    blobs, bounds = [], []
    for img, boundary in samples:
        ok, buf = cv2.imencode(".png", np.asarray(img))
        if not ok:
            raise ValueError("could not encode a sample frame as PNG")
        blobs.append(buf.tobytes())
        bounds.append(np.asarray(boundary, dtype=np.float64))
    np.savez_compressed(str(path), n=np.array(len(blobs)),
                        blobs=np.array(blobs, dtype=object),
                        bounds=np.stack(bounds) if bounds else np.zeros((0, 0)))
    return str(path)


def samples_from_npz(path):
    """Read back what `samples_to_npz` wrote, exactly."""
    z = np.load(str(path), allow_pickle=True)
    out = []
    for blob, bound in zip(z["blobs"], z["bounds"]):
        img = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        out.append((img, np.asarray(bound, dtype=np.float64)))
    return out


def iter_frames(video, every: int = 15, limit: int | None = None):
    """Yield (index, BGR frame) sampled across a clip, for the labelling window."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield i, frame
            if limit is not None and i // every >= limit:
                break
        i += 1
    cap.release()


def label_clip(video, every: int = 15, limit: int | None = None,
               scale: float = 1.0, on_frame=None, on_root=None):
    """Walk a clip, collecting a boundary per frame by clicking.

    A thin shell over `boundary_from_points`: the window collects points and
    everything testable is above. Keys: left click adds a point, `u` undo,
    return accept, `s` skip, escape quit.

    THE WINDOW IS TKINTER, NOT OPENCV. The obvious choice -- `cv2.namedWindow` --
    does not work here: this project's OpenCV is a HEADLESS build, and calling it
    fails with "the function is not implemented. Rebuild the library with
    Windows, GTK+ or Cocoa support", which says nothing about what to do. Tk is in
    the standard library, works on this display, and needs Pillow only to turn an
    array into something a canvas can show.

    `on_frame(index, frame, points, width, height)` may return a replacement
    frame for display, which is the hook a proposed boundary would use to draw
    itself for correction. Label collection is unaffected by it.

    `on_root(root)` is handed the Tk window once the handlers are bound. It
    exists so this can be driven without a human: a test schedules synthetic
    clicks and key presses through it, which is the only way to know the window
    works at all on a machine where nobody is going to click it.
    """
    import tkinter as tk

    from PIL import Image, ImageTk

    samples: list[tuple[np.ndarray, np.ndarray]] = []
    state = {"pts": [], "idx": 0, "frame": None, "done": False}

    root = tk.Tk()
    root.title("steadycut labeller")
    canvas = tk.Canvas(root, bg="black")
    canvas.pack()
    info = tk.Label(root, text="", anchor="w")
    info.pack(fill="x")

    def redraw():
        frame = state["frame"]
        h, w = frame.shape[:2]
        shown = frame
        if on_frame is not None:
            shown = on_frame(state["idx"], frame, list(state["pts"]), w, h)
        if scale != 1.0:
            shown = cv2.resize(shown, (int(w * scale), int(h * scale)))
        photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)))
        canvas.configure(width=photo.width(), height=photo.height())
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        sx = scale
        for px, py in state["pts"]:
            canvas.create_oval(px * sx - 4, py * sx - 4, px * sx + 4,
                               py * sx + 4, outline="red", width=2)
        if len(state["pts"]) >= 2:
            ordered = sorted(state["pts"])
            flat = [c for px, py in ordered for c in (px * sx, py * sx)]
            canvas.create_line(*flat, fill="lime", width=2)
        canvas._photo = photo            # keep a reference alive
        info.configure(
            text=f"frame {state['idx']}   {len(state['pts'])} points   "
                 f"click=add  u=undo  Return=accept  s=skip  Esc=quit")

    def on_click(event):
        state["pts"].append((int(event.x / scale), int(event.y / scale)))
        redraw()

    def finish(accept: bool):
        frame = state["frame"]
        w, h = frame.shape[1], frame.shape[0]
        if accept:
            b = boundary_from_points(state["pts"], w, h)
            if usable(b):
                samples.append((frame.copy(), b))
            else:
                print(f"  frame {state['idx']}: needs 2+ clicks, not labelled")
        state["pts"] = []
        if not load_next():
            root.quit()

    # ONE generator, pulled from: calling iter_frames() again would restart the
    # clip from frame 0 every time a frame was accepted.
    frames = iter_frames(video, every=every, limit=limit)

    def load_next() -> bool:
        try:
            idx, frame = next(frames)
        except StopIteration:
            state["done"] = True
            return False
        state["idx"], state["frame"] = idx, frame
        state["pts"] = []
        redraw()
        return True

    def on_key(event):
        k = (event.keysym or "").lower()
        if k == "escape":
            state["done"] = True
            root.quit()
        elif k == "u":
            if state["pts"]:
                state["pts"].pop()
                redraw()
        elif k == "return" or event.keysym == "KP_Enter":
            finish(True)
        elif k == "s":
            finish(False)

    canvas.bind("<Button-1>", on_click)
    root.bind("<Key>", on_key)
    root.bind("<Escape>", on_key)
    root.bind("<Return>", on_key)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    if on_root is not None:
        on_root(root, canvas, state)
    if load_next():
        # mainloop, not wait_window. `wait_window` waits for the window to be
        # DESTROYED, and `quit()` does not do that, so the first version hung
        # forever after the last frame -- with no error, and no way to tell from
        # the outside that it had stopped.
        root.mainloop()
    root.destroy()
    return samples


def store_refusal(report) -> str | None:
    """Why this fit must not be stored, or None when it may be.

    A fit that predicts nothing finite on held-out frames (NaN error) is
    degenerate -- usually too few frames -- and storing it poisons the
    loader: a bias-only cue refuses everything while LOOKING fitted.
    (Precedent on record: a 7-frame NaN-error fit stored Oct 1 that the
    loader preferred over the good 19-frame fit by content hash.)
    """
    err = report["held_out_error"]
    if err != err:  # NaN
        return (f"held-out error is NaN ({report['frames']} frames, "
                f"{report['train']} train) -- the cue predicts nothing "
                f"finite, label more frames")
    return None


def fit_samples(samples, source=None, held_out: float = 0.2, seed: int = 0):
    """Fit a `GroundCue` from labelled pairs and return (cue, report).

    The held-out fraction is real: the cue's docstring quotes a held-out error
    and that number only means something if some frames were kept back. With too
    few samples there is nothing to hold out, and the report says so rather than
    quoting a training error as if it were generalisation.
    """
    usable_samples = [(im, b) for im, b in samples if usable(b)]
    if len(usable_samples) < 3:
        raise ValueError(
            f"only {len(usable_samples)} usable labelled frames; a per-recording "
            f"fit needs at least 3 and gets unreliable fast below about 8")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(usable_samples))
    n_hold = max(1, int(round(len(usable_samples) * held_out)))
    hold, train = idx[:n_hold], idx[n_hold:]
    cue = GroundCue.fit([usable_samples[i] for i in train])
    err = float("nan")
    if len(hold):
        errs = []
        for i in hold:
            pred = np.asarray(cue.boundary(usable_samples[i][0]), dtype=float)
            truth = usable_samples[i][1]
            m = np.isfinite(pred) & np.isfinite(truth)
            if m.any():
                errs.append(np.abs(pred[m] - truth[m]).mean())
        if errs:
            err = float(np.mean(errs))
    report = {"frames": len(usable_samples), "train": len(train),
              "held_out": len(hold), "held_out_error": err}
    return cue, report


def write_profile(profile_path, cue, report, source=None) -> None:
    """Store the fitted cue on a `Profile`, recording what it was fitted from.

    `Profile.record_source` is what makes the staleness guard mean anything: a
    cue fitted on one export must not be applied to a re-export, and the only
    way that is enforceable is if the writing side records the claim. Calling it
    here rather than trusting a caller to remember is the whole point.
    """
    from steadycut.core.profiles import Profile

    path = Path(profile_path).expanduser()
    prof = Profile.from_dict(json.loads(path.read_text())) if path.exists() \
        else Profile()
    prof.cue = cue
    prof.fitted_on = report
    if source is not None:
        prof.record_source(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prof.to_dict(), indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="steadycut.framing.labeller",
        description="Label ground boundaries and fit a per-recording cue.")
    ap.add_argument("video", nargs="?", type=Path, help="clip to label")
    ap.add_argument("--samples", type=Path, help="write/read labelled samples .npz")
    ap.add_argument("--fit", type=Path, nargs="+",
                    help="fit a cue from one or more sample .npz files. The cue "
                         "is per RECORDING, so pass every window you labelled "
                         "and they are concatenated into one fit")
    ap.add_argument("--source", type=Path,
                    help="the source file the cue is fitted on (for the guard)")
    ap.add_argument("--profile", type=Path, help="profile JSON to write")
    ap.add_argument("--every", type=int, default=15, help="sample every N frames")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    args = ap.parse_args(argv)

    if args.fit:
        paths = args.fit if isinstance(args.fit, list) else [args.fit]
        # Concatenate across windows: the cue is fitted per RECORDING, not per
        # clip, and a recording has several windows with different lighting and
        # terrain. Fitting one window would learn that window.
        samples = []
        for p in paths:
            part = samples_from_npz(p)
            print(f"  {p}: {len(part)} labelled frames")
            samples.extend(part)
        cue, report = fit_samples(samples, source=args.source)
        reason = store_refusal(report)
        if reason is not None:
            print(f"  REFUSING to store: {reason}")
            return 2
        err = report["held_out_error"]
        if report["frames"] < 8:
            print(f"  WARNING: only {report['frames']} frames (unreliable "
                  f"below about 8) -- stored, but verify the held-out error "
                  f"{err:.4f} before trusting this cue")
        print(f"fitted on {report['frames']} frames from {len(paths)} file(s) "
              f"({report['train']} train, {report['held_out']} held out); "
              f"held-out error {report['held_out_error']:.4f} of frame height")
        if args.source is None:
            print("  note: no --source given, so nothing is recorded about what "
                  "this was fitted on and the staleness guard stays unchecked")
        if args.profile:
            write_profile(args.profile, cue, report, source=args.source)
            print(f"  wrote {args.profile}")
        return 0

    if not args.video:
        ap.error("give a VIDEO to label, or --fit to fit an existing sample set")
    samples = label_clip(args.video, every=args.every, limit=args.limit)
    if not args.samples:
        ap.error("--samples is required when labelling")
    samples_to_npz(samples, args.samples)
    usable_n = sum(1 for _, b in samples if usable(b))
    print(f"labelled {usable_n} of {len(samples)} shown frames -> {args.samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
