"""Two-pass blend stitch: the production stitch path.

Single-pass dfisheye draws the lens overlap twice with no blend (hard blue
core line, tone step, fringe 10-16 px on the T0 scoreboard). This module
stitches each lens to equirect separately -- with its own calibrated FOV,
per-clip gain match and per-body blue deflare -- and merges with a cached
ramp mask. Scoreboard (ski112): boundary step 22 -> 10 abs, fringe peak
35 -> 19, blue cast gone, remaining artifact a hairline achromatic cut.

Costs nothing when unused: every value here is generated on demand and
cached under ~/.cache/steadycut/stitch/, and the legacy single-pass path
stays available via StitchSpec(method="legacy").
"""
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Per-lens full FOVs from the X4 makernotes (182.15 / 180.58), the values the
# FOVxramp bench settled: nominal per-lens FOV, not the 186 deg corridor
# (wider buys blend width with rim pixels whose flare dominates the veil).
LENS_FOV_A = 182.2
LENS_FOV_B = 180.6
# Blend ramp matched to the measured overlap (~8 px at nominal FOV): wide
# enough to dissolve the cut, narrow enough to starve the veil.
RAMP_DEG = 4.0
# Blue deflare polynomial, B excess vs normalized lens radius, fitted on
# smooth-snow B-G (+6 center -> +13 rim, both lenses, R-G flat). Subtracted
# from the blue channel pre-stitch. Ceiling x1.0: x1.6 shows over-subtraction
# banding + top plug by eye, so this is not a tuning knob.
DEFLARE_POLY = (5.39, -7.2, 9.36, -0.85)
# Rim-median gain default (lens B toward A, measured ski112). Per-clip
# re-measurement is the production rule (auto-exposure moves); these are the
# fallback when measurement fails, never silently carried as truth -- the
# caller records which was used in the certificate.
GAIN_DEFAULT = (1.016, 1.029, 1.043)  # B, G, R multipliers for lens B


def cache_dir() -> Path:
    d = Path.home() / ".cache" / "steadycut" / "stitch"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class StitchSpec:
    """How the sphere is assembled. method="blend" is production."""
    # "legacy" is the default for cost reasons: the blend path bakes both
    # 2880^2 lens tracks per window (~20 min measured on a 6 s window at
    # 10 cores), so it is opt-in until that bake is optimized.
    method: str = "legacy"
    lens_fov_a: float = LENS_FOV_A
    lens_fov_b: float = LENS_FOV_B
    ramp_deg: float = RAMP_DEG
    deflare_poly: tuple = DEFLARE_POLY
    gain: tuple | None = None  # None = measure per clip; else (B,G,R) + flag
    optics_id: str | None = None  # registry optics profile serving this render
    leveling: tuple | None = None  # (pos_frame, neg_frame) seam softener or None


def mask_path(lens_fov_a: float, ramp_deg: float,
              w: int = 2048, h: int = 1024, fov_b: float | None = None) -> Path:
    """Cached blend ramp mask: 255 at sphere front, 0 at sphere back.

    Generated analytically from sphere yaw (sphere_maps.blend_weight), NOT
    with the old geq radial ramp, which wandered ~220 px off-meridian by row
    256. Same polarity as the retired bytes and the same serving graph; new
    cache key (v2).

    Polarity, carefully: in sphere_filtergraph the B track carries yaw=180,
    so B's lens center sits at x=0 and A's at x=1024; maskedmerge white
    selects the SECOND input. The PNG is therefore 255 where the graph shows
    B (front) and 0 where it shows A — proven by the delivered blend renders
    and by the 7.2-vs-76.5 inverted-merge probe, not by the filter docs.
    """
    from steadycut.render.sphere_maps import MASK_VERSION, blend_weight
    fb = fov_b if fov_b is not None else LENS_FOV_B
    dst = (cache_dir() /
           f"mask_{MASK_VERSION}_F{lens_fov_a:g}-{fb:g}_R{ramp_deg:g}_{w}x{h}.png")
    if dst.exists():
        return dst
    import numpy as np
    wgt, _ = blend_weight(w, h, ramp_deg, lens_fov_a, fb)
    cv2 = __import__("cv2")
    cv2.imwrite(str(dst), (wgt * 255).astype(np.uint8))
    return dst


def deflare_frame(lens_fov: float, poly: tuple,
                  size: int = 2880, center: float = 1440.0) -> Path:
    """Cached static blue-excess field for one lens: blend-subtracted fast.

    geq per frame costs ~1 fps; a static correction frame subtracted with the
    SIMD blend filter costs nothing per frame. Generated with numpy (exact
    polynomial, no geq parsing risk), cached per body curve.
    """
    import numpy as np
    key = "_".join(f"{c:g}" for c in poly)
    dst = cache_dir() / f"deflare_{key}_{size}.png"
    if dst.exists():
        return dst
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - center, yy - center) / center
    excess = np.polyval(np.array(poly), np.clip(r, 0, 1.05))
    cv2 = __import__("cv2")
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :, 0] = np.clip(excess, 0, 255).astype(np.uint8)
    cv2.imwrite(str(dst), img)
    return dst


def measure_gain(source: str, t: float = 5.0) -> tuple:
    """Per-clip rim-median gain (lens B toward A), cached per file.

    Auto-exposure moves per clip, so this is measured, not stored: one frame
    per track, median over the rim annulus (the only world both lenses
    share), B/A per channel. Falls back to GAIN_DEFAULT on any failure, and
    the caller must record which in the certificate.
    """
    import numpy as np
    key_src = f"{Path(source).resolve()}|{Path(source).stat().st_size}"
    cache = cache_dir() / "gains.json"
    try:
        db = json.loads(cache.read_text()) if cache.exists() else {}
        if key_src in db:
            return tuple(db[key_src])
    except (OSError, ValueError):
        db = {}
    cv2 = __import__("cv2")
    gains = GAIN_DEFAULT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for trk, tag in ((0, "A"), (1, "B")):
                subprocess.run(
                    f"ffmpeg -hide_banner -v error -ss {t} -i {source} "
                    f"-map 0:v:{trk} -frames:v 1 -y {tmp}/{tag}.png",
                    shell=True, check=True, timeout=120)
            yy, xx = np.mgrid[0:2880, 0:2880]
            r = np.hypot(xx - 1440, yy - 1440)
            m = (r >= 1350) & (r < 1440)
            meds = []
            for tag in ("A", "B"):
                img = cv2.imread(f"{tmp}/{tag}.png").astype(np.float32)
                meds.append(np.median(img[m].reshape(-1, 3), axis=0))
            a, b = meds
            gains = tuple(float(x) for x in a / np.maximum(b, 1))
            gains = tuple(min(max(g, 0.9), 1.1) for g in gains)  # sanity
            db[key_src] = list(gains)
            cache.write_text(json.dumps(db))
    except Exception:
        pass
    return gains


def bake_lens_tracks(source: str, spec: StitchSpec, gain: tuple,
                     start: float | None, duration: float | None,
                     workdir: Path, fps: str = "30000/1001") -> tuple[Path, Path]:
    """Bake deflared + gain-matched lens tracks to temp mp4s (one pass).

    Does the expensive per-lens work (blue-deflare subtract, gain match)
    OUTSIDE the sphere graph; see sphere_filtergraph for why that matters.
    Two outputs from one decode via split, so the source is read once.
    """
    db = deflare_frame(spec.lens_fov_a, spec.deflare_poly)
    gb, gg, gr = gain
    a_out, b_out = workdir / "lensA.mp4", workdir / "lensB.mp4"

    # Frame cap, not -t: an input-side -t was observed to be ignored (5568
    # frames for a 3 s window), so bound each output by frame count instead.
    import math
    nframes = None
    if duration is not None:
        try:
            from steadycut.render.reframe import source_fps as _fps
            nframes = math.ceil(duration * (_fps(source) or 29.97)) + 5
        except Exception:
            nframes = None
    # [1:v] is consumed by TWO blends, so it must be split: a filter output
    # can feed exactly ONE link.
    # setpts rebase is REQUIRED, not cosmetic: -ss as an input option leaves
    # the original timestamps (~110 s) on the baked frames, while the ramp
    # mask (looped still) and the sendcmd file both live at 0. The framesync
    # inside maskedmerge then starves (a 3 s window burned 200+ CPU-min with
    # zero bytes out). PTS-STARTPTS puts every input on the same clock.
    lv = ""
    if spec.leveling is not None:
        # Seam softener on track B (deflare precedent for asymmetric): add
        # the positive lobe, subtract the negative, around the gain stage.
        lv = ("[b0][lvP]blend=all_mode=addition,format=yuv444p[bx];"
              "[bx][lvN]blend=all_mode=subtract,format=yuv444p[by];")
    fc = (
        f"[0:v:0]format=gbrp[a0];[1:v]format=gbrp,split[df1][df2];"
        + (f"[2:v]format=gbrp[lvP];[3:v]format=gbrp[lvN];"
           if spec.leveling is not None else "") +
        f"[a0][df1]blend=all_mode=subtract,format=yuv444p,"
        f"setpts=PTS-STARTPTS[a];"
        f"[0:v:1]format=gbrp,lutrgb=r='clip(val*{gr:.4f},0,255)'"
        f":g='clip(val*{gg:.4f},0,255)':b='clip(val*{gb:.4f},0,255)'[b0];"
        + (f"{lv}[by][df2]blend=all_mode=subtract,format=yuv444p,"
           f"setpts=PTS-STARTPTS[b]"
           if spec.leveling is not None else
           f"[b0][df2]blend=all_mode=subtract,format=yuv444p,"
           f"setpts=PTS-STARTPTS[b]"))
    args = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    # -ss/-t are INPUT options: they must sit immediately before the source
    # they bound, or ffmpeg applies -t to the still-image input instead.
    if start is not None:
        args += ["-ss", f"{start:g}"]
    if duration is not None:
        args += ["-t", f"{duration:g}"]
    args += ["-i", str(source)]
    args += ["-framerate", fps, "-loop", "1", "-i", str(db)]
    if spec.leveling is not None:
        # Looped stills pinned to the video rate (same churn trap as the
        # deflare input); setpts rebase downstream puts them on clock 0.
        args += ["-framerate", fps, "-loop", "1", "-i", str(spec.leveling[0]),
                 "-framerate", fps, "-loop", "1", "-i", str(spec.leveling[1])]
    args += ["-filter_complex", fc,
             "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "16", "-pix_fmt", "yuv444p"]
    if nframes is not None:
        args += ["-frames:v", str(nframes)]
    args += [str(a_out),
             "-map", "[b]", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "16", "-pix_fmt", "yuv444p"]
    if nframes is not None:
        args += ["-frames:v", str(nframes)]
    args += [str(b_out)]
    subprocess.run(args, check=True)
    return a_out, b_out


def sphere_still(source: str, spec: StitchSpec, t: float = 5.0,
                 gain: tuple | None = None,
                 sphere_size=(2048, 1024)) -> Path:
    """One blend-graph equirect still (review/instrument use, not delivery).

    Per-lens v360 fisheye->equirect with each lens's own FOV (the same two
    passes sphere_filtergraph runs), deflare + gain applied like the bake,
    joined by the analytic mask. reframe emits flat only, so scoreboard and
    eye work that needs the sphere comes here.
    """
    import numpy as np
    from steadycut.render.sphere_maps import blend_weight
    cv2 = __import__("cv2")
    w, h = sphere_size
    g = gain or GAIN_DEFAULT
    gb, gg, gr = g
    with tempfile.TemporaryDirectory() as tmp:
        for trk in (0, 1):
            out = f"{tmp}/lens{trk}.png"
            subprocess.run(
                f"ffmpeg -hide_banner -v error -ss {t} -i {source} "
                f"-map 0:v:{trk} -frames:v 1 -y {out}",
                shell=True, check=True, timeout=120)
        a = cv2.imread(f"{tmp}/lens0.png")
        b = cv2.imread(f"{tmp}/lens1.png")
        # deflare (blue subtract) + gain, mirroring bake_lens_tracks.
        # The field stores excess in the BLUE channel: read color and take
        # [:,:,0], never grayscale (0.114*B rounds single-digit excess to 0).
        db = cv2.imread(str(deflare_frame(spec.lens_fov_a,
                                          spec.deflare_poly)))[:, :, 0]
        if db.shape[:2] != a.shape[:2]:
            db = cv2.resize(db, (a.shape[1], a.shape[0]))
        a = a.astype(np.float32)
        a[:, :, 0] = np.clip(a[:, :, 0] - db.astype(np.float32), 0, 255)
        b = np.clip(b.astype(np.float32) *
                    np.array(g, np.float32), 0, 255)  # g is (B,G,R), BGR order
        cv2.imwrite(f"{tmp}/pa.png", a.astype(np.uint8))
        cv2.imwrite(f"{tmp}/pb.png", b.astype(np.uint8))
        eq = []
        for tag, fov, yaw in (("pa", spec.lens_fov_a, 0),
                              ("pb", spec.lens_fov_b, 180)):
            p = f"{tmp}/eq_{tag}.png"
            eq.append(p)
            subprocess.run(
                f"ffmpeg -hide_banner -v error -i {tmp}/{tag}.png -vf "
                f"\"v360=input=fisheye:output=e:ih_fov={fov}:iv_fov={fov}"
                f":yaw={yaw}:w={w}:h={h}\" -frames:v 1 -y {p}",
                shell=True, check=True, timeout=120)
        ea = cv2.imread(eq[0]).astype(np.float32)
        eb = cv2.imread(eq[1]).astype(np.float32)
        wgt, _ = blend_weight(w, h, spec.ramp_deg, spec.lens_fov_a)
        w3 = wgt[:, :, None]
        if ea.shape[:2] != (h, w):
            ea = cv2.resize(ea, (w, h))
        if eb.shape[:2] != (h, w):
            eb = cv2.resize(eb, (w, h))
        # Merge polarity mirrors the serving graph: maskedmerge white selects
        # the SECOND input ([c][d][e] = A, B, mask, so 255 shows B). The mask
        # PNG is 255 at sphere front (x=0) where B's yaw-180 equirect carries
        # the content; wgt is 1.0 on A's sphere hemisphere. Both conventions
        # are pinned in test_sphere_maps/test_stitch — touch neither blindly.
        img = np.clip(eb * w3 + ea * (1 - w3), 0, 255).astype(np.uint8)
        dst = (cache_dir() /
               f"still_{Path(source).stem}_t{t:g}_"
               f"F{spec.lens_fov_a:g}-{spec.lens_fov_b:g}.png")
        cv2.imwrite(str(dst), img)
        return dst


def measure_step(still_path, width=2048, seams=(512, 1536),
                 plateau=(10, 40)) -> dict:
    """Tone step across each seam: right-plateau median minus left, per
    channel (B,G,R signed). Plateaus avoid the ramp core."""
    import numpy as np
    cv2 = __import__("cv2")
    img = cv2.imread(str(still_path)).astype(np.float32)
    h, w = img.shape[:2]
    out = {}
    for s in seams:
        x = int(s * w / width)
        p0a, p0b, p1a, p1b = x - plateau[1], x - plateau[0], x + plateau[0], \
            x + plateau[1]
        left = np.median(img[:, p0a:p0b].reshape(-1, 3), axis=0)
        right = np.median(img[:, p1a:p1b].reshape(-1, 3), axis=0)
        out[s] = [round(float(v), 2) for v in (right - left)]
    return out


STEP_LIMIT_DN = 8.0  # larger means geometry is wrong: fall back, don't paint


def leveling_frames(spec: StitchSpec, steps: dict, size=2880, w=2048,
                    feather_px=48) -> tuple:
    """Seam-ramp correction frames for the lens-B track (deflare-style).

    steps: {seam_x: [dB, dG, dR]} in sphere pixels at width w. The full step
    rides on track B (asymmetric, deflare precedent): corr = s*f, f a cosine
    falloff peaking at the seam, split into pos/neg unsigned frames for the
    add/subtract blends. Returns (pos_path, neg_path, cert); steps above
    STEP_LIMIT_DN fall back (no frames + flag, never paint broken geometry).
    seam_xs may be a per-row polyline array (routed seams, Phase 3).
    """
    import numpy as np
    from steadycut.render.sphere_maps import lens_to_sphere
    cv2 = __import__("cv2")
    cert = {"seams": {str(k): v for k, v in steps.items()}}
    worst = max(abs(v) for s in steps.values() for v in s)
    if worst > STEP_LIMIT_DN:
        cert.update({"applied": False,
                     "reason": f"step {worst:.1f} DN > {STEP_LIMIT_DN} DN: "
                               "geometry suspect, leveling refused"})
        return None, None, cert
    sx, _, valid = lens_to_sphere(spec.lens_fov_b, size=size, w=w,
                                  h=w // 2, back=True)
    # Per-seam feathers (never one combined f: steps differ per seam and a
    # shared feather mixes them — measured: 1536's +6 leaked into 512's -5).
    # Factor 1, deliberately NOT 2: a feather-localized op softens the seam-
    # center hairline but cannot move plateaus (and symmetric ± cancels at
    # w=0.5). Plateau steps belong to per-clip gain matching; this only
    # takes the edge off the center. f peaks 1.0 at the seam.
    corr = np.zeros((size, size, 3), np.float32)
    for s, st in steps.items():
        if np.ndim(s) == 0:
            d = np.abs(sx - float(s) * w / 2048)
        else:
            arr = np.asarray(s, dtype=float)
            rows = np.clip((np.arange(sx.shape[0]) * len(arr)
                            // sx.shape[0]), 0, len(arr) - 1)
            d = np.abs(sx - arr[rows][:, None] * w / 2048)
        d = np.minimum(d, w - d)  # wrap at the x=0 meridian
        f = np.where(d < feather_px,
                     0.5 * (1 + np.cos(np.pi * d / feather_px)), 0.0)
        f = np.where(valid, f, 0.0)
        corr += np.array(st, np.float32) * f[:, :, None]
    pos = np.clip(corr, 0, 255).astype(np.uint8)
    neg = np.clip(-corr, 0, 255).astype(np.uint8)
    key = (f"{spec.lens_fov_b:g}_{len(steps)}seam_"
           + "_".join(f"{v:+.1f}" for s in steps.values() for v in s))
    pp = cache_dir() / f"level_pos_{key}_{size}.png"
    np_ = cache_dir() / f"level_neg_{key}_{size}.png"
    cv2.imwrite(str(pp), pos)
    cv2.imwrite(str(np_), neg)
    cert.update({"applied": True, "worst_dn": round(float(worst), 2)})
    return pp, np_, cert


def sphere_filtergraph(spec: StitchSpec, gain: tuple,
                       sphere_size=(2048, 1024)) -> tuple[str, Path]:
    """Two-pass sphere assembly over BAKED lens tracks (see bake_lens_tracks:
    deflare + gain already applied). Inputs: [0:v] lens A equirect source,
    [1:v] lens B equirect source, [2:v] ramp mask. Each lens is projected
    with its own calibrated FOV; maskedmerge joins them on [s].
    Returns (filter snippet, mask path)."""
    w, h = sphere_size
    mask = mask_path(spec.lens_fov_a, spec.ramp_deg, w, h, spec.lens_fov_b)
    gb, gg, gr = gain
    fg = (
        f"[0:v]v360=input=fisheye:output=e:ih_fov={spec.lens_fov_a}"
        f":iv_fov={spec.lens_fov_a}:w={w}:h={h},format=yuv444p[c];"
        f"[1:v]v360=input=fisheye:output=e:ih_fov={spec.lens_fov_b}"
        f":iv_fov={spec.lens_fov_b}:yaw=180:w={w}:h={h},format=yuv444p[d];"
        f"[2:v]format=gbrp[e];[c][d][e]maskedmerge,format=yuv444p[s]")
    return fg, mask
