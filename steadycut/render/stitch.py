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
    deflare_scale: tuple | None = None  # None = measure per clip (A, B);
    # the static poly fits one sun, per-lens scale fits this one
    gain: tuple | None = None  # None = measure per clip; else (B,G,R) + flag
    optics_id: str | None = None  # registry optics profile serving this render
    leveling: tuple | None = None  # (pos_frame, neg_frame) seam softener or None
    seam: str = "meridian"  # or "routed": Viterbi polyline mask
    seam_path: str | None = None  # npz {seam_x: polyline} when seam=routed


def mask_path(lens_fov_a: float, ramp_deg: float,
              w: int = 2048, h: int = 1024, fov_b: float | None = None,
              polylines: dict | None = None) -> Path:
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
    import numpy as np
    fb = fov_b if fov_b is not None else LENS_FOV_B
    tag = ""
    if polylines is not None:
        from steadycut.render.sphere_maps import routed_weight
        import hashlib
        key = hashlib.sha1(np.concatenate(
            [np.asarray(v, dtype=np.float32).ravel()
             for _, v in sorted(polylines.items())]).tobytes()).hexdigest()[:12]
        tag = f"_r{key}"
    dst = (cache_dir() /
           f"mask_{MASK_VERSION}_F{lens_fov_a:g}-{fb:g}_R{ramp_deg:g}_{w}x{h}{tag}.png")
    if dst.exists():
        return dst
    if polylines is None:
        wgt, _ = blend_weight(w, h, ramp_deg, lens_fov_a, fb)
    else:
        wgt, _ = routed_weight(w, h, ramp_deg, lens_fov_a, fb,
                               polylines=polylines)
    cv2 = __import__("cv2")
    cv2.imwrite(str(dst), (wgt * 255).astype(np.uint8))
    return dst


def deflare_frame(lens_fov: float, poly: tuple, scale=1.0,
                  size: int = 2880, center: float = 1440.0) -> Path:
    """Cached blue-excess field for one lens: blend-subtracted fast.

    geq per frame costs ~1 fps; a static correction frame subtracted with the
    SIMD blend filter costs nothing per frame. Generated with numpy (exact
    polynomial, no geq parsing risk), cached per body curve. scale is the
    per-clip flare (measure_deflare_scale): float, or a 10-list of
    per-radial-bin scales (this sun's shape, not just its magnitude).
    Cached per curve+scale.
    """
    import numpy as np
    import json as _json
    key = ("_".join(f"{c:g}" for c in poly) + "_x" +
           (_json.dumps(scale) if isinstance(scale, list)
            else f"{float(scale):.3f}"))
    dst = cache_dir() / f"deflare_{key}_{size}.png"
    if dst.exists():
        return dst
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - center, yy - center) / center
    base = np.polyval(np.array(poly), np.clip(r, 0, 1.05))
    if isinstance(scale, list):
        s = np.interp(np.clip(r, 0, 1.0) * 10 - 0.5, np.arange(10),
                      np.asarray(scale, float))
        excess = base * np.clip(s, 0, 2.5)
    else:
        excess = float(scale) * base
    cv2 = __import__("cv2")
    img = np.zeros((size, size, 3), np.uint8)
    img[:, :, 0] = np.clip(excess, 0, 255).astype(np.uint8)
    cv2.imwrite(str(dst), img)
    return dst


def measure_deflare_scale(source: str, poly: tuple, t: float = 5.0):
    """Per-lens flare magnitude (A, B) for the static deflare poly.

    The two lenses flare differently (Jan26: A a +3 white rim ring, B +22
    blue sky flare) — a shared scale fixes the average and overshoots the
    cleaner lens. Ratios measured blue-excess against the poly prediction
    per radial bin, median per lens. 1.0 when a lens shows nothing to
    scale. Clamped 0..2.5. Cached per file+t. Returns (scale_A, scale_B).

    Extracts both lens frames at the render time, takes flat bright pixels
    (sky/snow: bright + neutral, mid-field disk so scene gradients don't
    pose as flare), and ratios measured blue-excess (B minus neutral
    baseline R) against the poly prediction per radial bin. Median per
    lens over bins with real predicted flare; 1.0 when nothing is
    predicted and little is seen (no flare to scale). Clamped 0..2.5:
    a blind scale-up paints blue where the model has no support.
    """
    import numpy as np
    key_src = (f"{Path(source).resolve()}|{Path(source).stat().st_size}"
               f"|{round(t, 1)}|deflare4")
    cache = cache_dir() / "gains.json"
    try:
        db = json.loads(cache.read_text()) if cache.exists() else {}
        if key_src in db:
            v = db[key_src]
            return tuple(v) if isinstance(v, list) else (1.0, 1.0)
    except (OSError, ValueError):
        db = {}
    scale = (1.0, 1.0)
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for trk, tag in ((0, "A"), (1, "B")):
                subprocess.run(
                    f"ffmpeg -hide_banner -v error -ss {t} -i {source} "
                    f"-map 0:v:{trk} -frames:v 1 -y {tmp}/{tag}.png",
                    shell=True, check=True, timeout=120)
            cv2 = __import__("cv2")
            pred = np.polyval(np.array(poly, float),
                              np.clip(np.linspace(0, 1.0, 10), 0, 1.05))
            scales = []
            for tag in ("A", "B"):
                img = cv2.imread(f"{tmp}/{tag}.png").astype(np.float32)
                yy, xx = np.mgrid[0:2880, 0:2880]
                r = np.hypot(xx - 1440, yy - 1440) / 1440
                g = img.mean(axis=2)
                # rim-inclusive (r<1.0): the seam lives at the rim, and a
                # mid-field-only fit need not hold there (Jan26 snow-blue
                # at 1536 survived a mid-field scale). Bright gate drops
                # the black exterior; neutral gate drops trees/texture.
                flat = (r < 1.0) & (g > 150) & (
                    np.abs(img[:, :, 1] - img[:, :, 2]) < 15)
                # per-BIN scale (shape, not just magnitude): the static
                # poly's shape is one sun's; this sun differs by radius.
                # Rim bins first (the seam lives at r~0.94); fall back
                # inward only when the rim is empty. Smoothed over radius
                # (noisy bins must not whip the correction).
                bin_s = np.full(10, np.nan)
                if flat.mean() >= 0.05:
                    exc = img[:, :, 0] - img[:, :, 2]
                    for rmin in (0.8, 0.0):
                        for i in range(10):
                            if i / 10 < rmin or not np.isnan(bin_s[i]):
                                continue
                            m = flat & (r >= i / 10) & (r < (i + 1) / 10)
                            if m.sum() < 2000 or pred[i] < 3.0:
                                continue
                            got = float(np.median(exc[m]))
                            if got > 1.0:
                                bin_s[i] = min(max(got / pred[i], 0.0),
                                               2.5)
                        if not np.isnan(bin_s[8:]).all():
                            break
                if np.isnan(bin_s).all():
                    scales.append(1.0)
                    continue
                # fill gaps from neighbors, smooth, fall back to 1.0 where
                # the model has no support
                xs10 = np.arange(10)
                ok = ~np.isnan(bin_s)
                filled = np.interp(xs10, xs10[ok], bin_s[ok],
                                   left=bin_s[ok][0], right=bin_s[ok][-1])
                sm = np.convolve(filled, [0.25, 0.5, 0.25], mode="same")
                scales.append(sm.tolist())
            scale = (scales[0], scales[1])
            db[key_src] = list(scale)
            cache.write_text(json.dumps(db))
    except Exception:
        pass
    return scale if isinstance(scale, tuple) else (1.0, 1.0)


def measure_gain(source: str, t: float = 5.0) -> tuple:
    """Per-clip gain (lens B toward A), cached per file+time.

    Snow-first: median over neutral-bright mid-field pixels (the same
    illuminant as the seam content). Rim-annulus fallback when no snow:
    rim gain measured on edges hurts snow, and AE drifts across a clip,
    so the cache key carries the measurement time and callers pass the
    render time, not a fixed 5 s. Neutral scalar for snow (hue matches);
    per-channel rim legacy below. Falls back to GAIN_DEFAULT on failure.

    Auto-exposure moves per clip, so this is measured, not stored: one
    frame per track at the render time. The caller records which in the
    certificate.
    """
    import numpy as np
    key_src = (f"{Path(source).resolve()}|{Path(source).stat().st_size}"
               f"|{round(t, 1)}")
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
            imgs = {tag: cv2.imread(f"{tmp}/{tag}.png").astype(np.float32)
                    for tag in ("A", "B")}
            # Snow first (mid-field, same illuminant as the seam content):
            # rim-annulus gain measured on edges actively hurts snow (Jan26:
            # rim 1.005 match, snow B/A 1.02 — the rim boost doubled the
            # snow step). Neutral scalar: snow hue already matches (B-R
            # within 2); per-channel risks the Feb21 green tint.
            snow = {}
            for tag, img in imgs.items():
                g = img.mean(axis=2)
                m = ((r < 1200) & (g > 150)
                     & (np.abs(img[:, :, 0] - img[:, :, 2]) < 20)
                     & (np.abs(img[:, :, 1] - img[:, :, 2]) < 20))
                snow[tag] = (m.mean(), np.median(img[m].reshape(-1, 3),
                                                 axis=0) if m.any() else None)
            if (snow["A"][1] is not None and snow["B"][1] is not None
                    and snow["A"][0] > 0.10 and snow["B"][0] > 0.10):
                g = float(snow["A"][1].mean() / snow["B"][1].mean())
                g = min(max(g, 0.9), 1.1)
                gains = (g, g, g)
            else:
                m = (r >= 1350) & (r < 1440)
                meds = []
                for tag in ("A", "B"):
                    meds.append(np.median(imgs[tag][m].reshape(-1, 3),
                                          axis=0))
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
    scl = spec.deflare_scale
    if scl is None:
        scl = measure_deflare_scale(source, spec.deflare_poly,
                                    start if start is not None else 5.0)
    if not isinstance(scl, (tuple, list)):
        scl = (float(scl), float(scl))
    dbA = deflare_frame(spec.lens_fov_a, spec.deflare_poly, scl[0])
    dbB = deflare_frame(spec.lens_fov_b, spec.deflare_poly, scl[1])
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
    # [1:v]/[2:v] are the per-lens deflare frames (different flare per
    # lens: a shared frame fixes the average and overshoots the cleaner
    # one). A filter output feeds exactly ONE link, so no split needed.
    # setpts rebase is REQUIRED, not cosmetic: -ss as an input option leaves
    # the original timestamps (~110 s) on the baked frames, while the ramp
    # mask (looped still) and the sendcmd file both live at 0. The framesync
    # inside maskedmerge then starves (a 3 s window burned 200+ CPU-min with
    # zero bytes out). PTS-STARTPTS puts every input on the same clock.
    lv = ""
    if spec.leveling is not None:
        # Seam softener on track B (deflare precedent for asymmetric): add
        # the positive lobe, subtract the negative, around the gain stage.
        # ALL blend inputs stay gbrp (the old deflare path's convention);
        # a single format=yuv444p closes the chain. Converting mid-chain
        # (format= inside a blend) feeds the next blend mismatched planes
        # and the output comes out monochrome green — measured, v3.
        lv = ("[b0][lvP]blend=all_mode=addition[bx];"
              "[bx][lvN]blend=all_mode=subtract[by];")
    fc = (
        f"[0:v:0]format=gbrp[a0];[1:v]format=gbrp[df1];[2:v]format=gbrp[df2];"
        + (f"[3:v]format=gbrp[lvP];[4:v]format=gbrp[lvN];"
           if spec.leveling is not None else "") +
        # fps AFTER setpts (CFR rebase): the two .insv streams have
        # independent clocks (PTS gaps in B); without this the downstream
        # maskedmerge framesync starves intermittently and the B half of
        # the delivered frame blinks black (measured 19.8 DN every few
        # frames). Filling by duplication, never by black.
        f"[a0][df1]blend=all_mode=subtract,format=yuv444p,"
        f"setpts=PTS-STARTPTS,fps={fps}[a];"
        f"[0:v:1]format=gbrp,lutrgb=r='clip(val*{gr:.4f},0,255)'"
        f":g='clip(val*{gg:.4f},0,255)':b='clip(val*{gb:.4f},0,255)'[b0];"
        + (f"{lv}[by][df2]blend=all_mode=subtract,format=yuv444p,"
           f"setpts=PTS-STARTPTS,fps={fps}[b]"
           if spec.leveling is not None else
           f"[b0][df2]blend=all_mode=subtract,format=yuv444p,"
           f"setpts=PTS-STARTPTS,fps={fps}[b]"))
    args = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    # -ss/-t are INPUT options: they must sit immediately before the source
    # they bound, or ffmpeg applies -t to the still-image input instead.
    if start is not None:
        args += ["-ss", f"{start:g}"]
    if duration is not None:
        args += ["-t", f"{duration:g}"]
    args += ["-i", str(source)]
    args += ["-framerate", fps, "-loop", "1", "-i", str(dbA),
             "-framerate", fps, "-loop", "1", "-i", str(dbB)]
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
                 sphere_size=(2048, 1024), keep_equirects_dir=None,
                 repair: bool = False, spans=None) -> Path:
    """One blend-graph equirect still (review/instrument use, not delivery).

    Per-lens v360 fisheye->equirect with each lens's own FOV (the same two
    passes sphere_filtergraph runs), deflare + gain applied like the bake,
    joined by the analytic mask. reframe emits flat only, so scoreboard and
    eye work that needs the sphere comes here. keep_equirects_dir: when set,
    the preprocessed per-lens equirects (A first, B second, production yaw
    convention) are kept there for flow/routing stages sharing the pixels.
    repair: run near-field auto repair (gated, cert-printed) on the merge.
    spans: explicit operator spans (nearfield.load_spans JSON) repaired
    instead of auto detection.
    """
    import numpy as np
    from steadycut.render.sphere_maps import blend_weight
    cv2 = __import__("cv2")
    w, h = sphere_size
    g = gain or measure_gain(source, t)  # stills match the bake
    with tempfile.TemporaryDirectory() as tmp:
        ea, eb = lens_equirects(source, spec, g, t, tmp, sphere_size,
                                keep_dir=keep_equirects_dir)
        wgt, _ = blend_weight(w, h, spec.ramp_deg, spec.lens_fov_a,
                                spec.lens_fov_b)
        if getattr(spec, "seam", "meridian") == "routed":
            # route on uncorrected pixels (texture cost is tone-blind),
            # then every stage below shares the routed weight
            from steadycut.render.seam_route import (route_both,
                                                    route_weight)
            if spec.seam_path:
                polys = load_polylines(spec.seam_path)
            else:
                polys = route_both(ea.astype(np.uint8),
                                   eb.astype(np.uint8))
                print(f"CLI| routed seams: " +
                      ", ".join(f"{s}:{np.median(p):.0f}" +
                                f"(range {p.min():.0f}-{p.max():.0f})"
                                for s, p in sorted(polys.items())))
            if polys:
                wgt = route_weight(polys, wgt)
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
        from steadycut.render.nearfield import (edge_flatten,
                                                eroded_validity,
                                                multiband_merge,
                                                residual_level)
        lumA = ea.mean(axis=2) if ea.ndim == 3 else ea
        lumB = eb.mean(axis=2) if eb.ndim == 3 else eb
        vA, vB = eroded_validity(lumA), eroded_validity(lumB)
        ea, eb, _einfo = edge_flatten(ea, eb, wgt, vA, vB)
        ea, eb, _linfo = residual_level(ea, eb, wgt, vA, vB)
        img = multiband_merge(ea.astype(np.uint8), eb.astype(np.uint8),
                              wgt, vA, vB)
        if spans or repair:
            from steadycut.render.nearfield import (auto_repair, load_spans,
                                                    repair_spans)
            if spans:
                parsed = load_spans(spans) if isinstance(spans, str) else spans
                img = repair_spans(ea.astype(np.uint8), eb.astype(np.uint8),
                                   wgt, parsed, spec.lens_fov_a,
                                   spec.lens_fov_b)
                print(f"CLI| nearfield {len(parsed)} explicit spans repaired")
            else:
                img, ncert = auto_repair(ea.astype(np.uint8),
                                         eb.astype(np.uint8), wgt,
                                         spec.lens_fov_a, spec.lens_fov_b)
                for s in ncert["spans"]:
                    print(f"CLI| nearfield seam {s['seam']} rect {s['rect']}"
                          f" dx {s['dx']} votes {s['votes']}: {s['action']}")
        dst = (cache_dir() /
               f"still_{Path(source).stem}_t{t:g}_"
               f"F{spec.lens_fov_a:g}-{spec.lens_fov_b:g}_R{spec.ramp_deg:g}"
               f"_S{getattr(spec, 'seam', 'meridian')}.png")
        cv2.imwrite(str(dst), img)
        return dst


def lens_equirects(source, spec, gain, t, tmp, sphere_size=(2048, 1024),
                   keep_dir=None):
    """Shared first half of sphere_still: per-lens extract, per-lens deflare,
    gain, rim, v360 to equirect. Returns (ea, eb) float32 BGR. tone_fields
    and sphere_still share it so video and stills correct identically.
    keep_dir: when set, the per-lens equirects are kept there (eqA/eqB)."""
    import numpy as np
    import subprocess
    cv2 = __import__("cv2")
    w, h = sphere_size
    for trk in (0, 1):
        out = f"{tmp}/lens{trk}.png"
        subprocess.run(
            f"ffmpeg -hide_banner -v error -ss {t} -i {source} "
            f"-map 0:v:{trk} -frames:v 1 -y {out}",
            shell=True, check=True, timeout=120)
    a = cv2.imread(f"{tmp}/lens0.png")
    b = cv2.imread(f"{tmp}/lens1.png")
    scl = spec.deflare_scale
    if scl is None:
        scl = measure_deflare_scale(source, spec.deflare_poly, t)
    if not isinstance(scl, (tuple, list)):
        scl = (float(scl), float(scl))
    dbA = cv2.imread(str(deflare_frame(
        spec.lens_fov_a, spec.deflare_poly, scl[0])))[:, :, 0]
    dbB = cv2.imread(str(deflare_frame(
        spec.lens_fov_b, spec.deflare_poly, scl[1])))[:, :, 0]
    if dbA.shape[:2] != a.shape[:2]:
        dbA = cv2.resize(dbA, (a.shape[1], a.shape[0]))
    if dbB.shape[:2] != a.shape[:2]:
        dbB = cv2.resize(dbB, (a.shape[1], a.shape[0]))
    a = a.astype(np.float32)
    a[:, :, 0] = np.clip(a[:, :, 0] - dbA.astype(np.float32), 0, 255)
    b = np.clip(b.astype(np.float32) *
                np.array(gain, np.float32), 0, 255)
    b[:, :, 0] = np.clip(b[:, :, 0] - dbB.astype(np.float32), 0, 255)
    from steadycut.render.rim import rim_blue_correct
    a = rim_blue_correct(a.astype(np.uint8)).astype(np.float32)
    b = rim_blue_correct(b.astype(np.uint8)).astype(np.float32)
    cv2.imwrite(f"{tmp}/pa.png", a.astype(np.uint8))
    cv2.imwrite(f"{tmp}/pb.png", b.astype(np.uint8))
    eq = []
    for tag, fov, yaw in (("pa", spec.lens_fov_a, 0),
                          ("pb", spec.lens_fov_b, 180)):
        q = f"{tmp}/eq_{tag}.png"
        eq.append(q)
        subprocess.run(
            f"ffmpeg -hide_banner -v error -i {tmp}/{tag}.png -vf "
            f"\"v360=input=fisheye:output=e:ih_fov={fov}:iv_fov={fov}"
            f":yaw={yaw}:w={w}:h={h}\" -frames:v 1 -y {q}",
            shell=True, check=True, timeout=120)
    ea = cv2.imread(eq[0]).astype(np.float32)
    eb = cv2.imread(eq[1]).astype(np.float32)
    if keep_dir is not None:
        import shutil
        from pathlib import Path as _P
        kd = _P(keep_dir)
        kd.mkdir(parents=True, exist_ok=True)
        shutil.copy(eq[0], kd / "eqA.png")
        shutil.copy(eq[1], kd / "eqB.png")
    return ea, eb



def tone_fields(source: str, spec, gain: tuple, t: float,
                sphere_size=(2048, 1024)):
    """Per-window static tone corrections for the VIDEO path (cached/file+t).

    The still path corrects per frame in numpy (edge_flatten +
    residual_level); video cannot afford per-frame Python, so the same two
    stages run ONCE on a representative frame and the corrections ride as
    four looped stills (pos/neg per lens: blend has no signed pixels),
    subtracted in sphere_filtergraph like the leveling lobes. AE drift
    across a window is small; static is honest. Returns
    (posA, negA, posB, negB) Paths + info dict.
    """
    import numpy as np
    import tempfile
    cv2 = __import__("cv2")
    key_src = (f"{Path(source).resolve()}|{Path(source).stat().st_size}"
               f"|{round(t, 1)}|tone")
    try:
        db = None
        cache = cache_dir() / "gains.json"
        db = json.loads(cache.read_text()) if cache.exists() else {}
        if key_src in db and all((cache_dir() / f).exists()
                                 for f in db[key_src]["files"]):
            return tuple(cache_dir() / f for f in db[key_src]["files"]), \
                db[key_src]["info"]
    except (OSError, ValueError):
        db = {}
    with tempfile.TemporaryDirectory() as tmp:
        ea, eb = lens_equirects(source, spec, gain, t, tmp, sphere_size)
        w, h = sphere_size
        from steadycut.render.sphere_maps import blend_weight
        from steadycut.render.nearfield import (edge_flatten, eroded_validity,
                                                residual_level)
        wgt, _ = blend_weight(w, h, spec.ramp_deg, spec.lens_fov_a,
                              spec.lens_fov_b)
        vA = eroded_validity(ea.mean(axis=2))
        vB = eroded_validity(eb.mean(axis=2))
        ea2, eb2, einfo = edge_flatten(ea.copy(), eb.copy(), wgt, vA, vB)
        ea3, eb3, linfo = residual_level(ea2, eb2, wgt, vA, vB)
        info = {"edge": einfo, "level": linfo}
        outs = []
        for tag, before, after in (("toneA", ea, ea3), ("toneB", eb, eb3)):
            corr = after - before
            for nm, part in (("pos", np.clip(corr, 0, 255)),
                             ("neg", np.clip(-corr, 0, 255))):
                p = (cache_dir() /
                     f"tone_{Path(source).stem}_t{t:g}_{tag}_{nm}.png")
                cv2.imwrite(str(p), part.astype(np.uint8))
                outs.append(p)
    try:
        if db is not None:
            db[key_src] = {"files": [f.name for f in outs], "info": info}
            (cache_dir() / "gains.json").write_text(json.dumps(db))
    except (OSError, ValueError):
        pass
    return tuple(outs), info


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


def sphere_delta_to_lens(spec: StitchSpec, delta, size=2880) -> tuple:
    """Map a sphere-space RGB correction delta onto the lens-B track.

    delta: float32 (h, w, 3) sphere still (e.g. flow-corrected B minus B).
    Inverse-samples through lens_to_sphere so the correction rides the
    existing bake blends (same mechanism as leveling) instead of requiring
    an own-remap sphere bake (Phase 6 verdict: unjustified). Returns
    (pos_frame, neg_frame) uint8 paths; zero delta yields empty frames.
    """
    import numpy as np
    from steadycut.render.sphere_maps import lens_to_sphere
    cv2 = __import__("cv2")
    dh, dw = delta.shape[:2]
    sx, sy, valid = lens_to_sphere(spec.lens_fov_b, size=size, w=dw, h=dh,
                                   back=True)
    lens = cv2.remap(delta.astype(np.float32), sx, sy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lens = np.where(valid[:, :, None], lens, 0.0)
    pos = np.clip(lens, 0, 255).astype(np.uint8)
    neg = np.clip(-lens, 0, 255).astype(np.uint8)
    key = f"{spec.lens_fov_b:g}_delta_{size}"
    pp = cache_dir() / f"flow_pos_{key}.png"
    nn = cache_dir() / f"flow_neg_{key}.png"
    cv2.imwrite(str(pp), pos)
    cv2.imwrite(str(nn), neg)
    return pp, nn


def save_polylines(polylines: dict) -> Path:
    """Cache Viterbi polylines under a content hash; spec carries the path."""
    import hashlib
    import numpy as np
    key = hashlib.sha1(np.concatenate(
        [np.asarray(v, dtype=np.float32).ravel()
         for _, v in sorted(polylines.items())]).tobytes()).hexdigest()[:12]
    dst = cache_dir() / f"seam_{key}.npz"
    if not dst.exists():
        np.savez(dst, **{str(k): np.asarray(v, dtype=np.float32)
                         for k, v in polylines.items()})
    return dst


def load_polylines(path) -> dict | None:
    import numpy as np
    try:
        z = np.load(str(path))
        return {int(k): np.asarray(z[k], dtype=float) for k in z.files}
    except Exception:
        return None


def sphere_filtergraph(spec: StitchSpec, gain: tuple,
                       sphere_size=(2048, 1024), tone=None) -> tuple[str, Path]:
    """Two-pass sphere assembly over BAKED lens tracks (see bake_lens_tracks:
    deflare + gain already applied). Inputs: [0:v] lens A equirect source,
    [1:v] lens B equirect source, [2:v] ramp mask. Each lens is projected
    with its own calibrated FOV; maskedmerge joins them on [s].
    tone: optional (posA, negA, posB, negB) looped-still Paths from
    tone_fields (static per-window edge/residual corrections). They ride as
    [3:v]..[6:v]; applied post-v360 pre-merge in GBRP (corrections are
    computed in BGR). Returns (filter snippet, mask path)."""
    w, h = sphere_size
    polys = None
    if getattr(spec, "seam", "meridian") == "routed" and spec.seam_path:
        polys = load_polylines(spec.seam_path)
    mask = mask_path(spec.lens_fov_a, spec.ramp_deg, w, h, spec.lens_fov_b,
                     polylines=polys)
    gb, gg, gr = gain
    fg = (
        f"[0:v]v360=input=fisheye:output=e:ih_fov={spec.lens_fov_a}"
        f":iv_fov={spec.lens_fov_a}:w={w}:h={h},format=yuv444p[c];"
        f"[1:v]v360=input=fisheye:output=e:ih_fov={spec.lens_fov_b}"
        f":iv_fov={spec.lens_fov_b}:yaw=180:w={w}:h={h},format=yuv444p[d];"
        f"[2:v]format=gbrp[e];[c][d][e]maskedmerge,format=yuv444p[s]")
    if tone is not None:
        # static tone corrections post-v360 (GBRP: corrections are BGR),
        # pre-merge: [3:v]..[6:v] = posA/negA/posB/negB looped stills
        fg = (
            f"[0:v]v360=input=fisheye:output=e:ih_fov={spec.lens_fov_a}"
            f":iv_fov={spec.lens_fov_a}:w={w}:h={h},format=gbrp[c0];"
            f"[c0][3:v]format=gbrp[tAp];[c0][tAp]blend=all_mode=addition[c1];"
            f"[c1][4:v]format=gbrp[tAn];[c1][tAn]blend=all_mode=subtract[c];"
            f"[1:v]v360=input=fisheye:output=e:ih_fov={spec.lens_fov_b}"
            f":iv_fov={spec.lens_fov_b}:yaw=180:w={w}:h={h},format=gbrp[d0];"
            f"[d0][5:v]format=gbrp[tBp];[d0][tBp]blend=all_mode=addition[d1];"
            f"[d1][6:v]format=gbrp[tBn];[d1][tBn]blend=all_mode=subtract[d];"
            f"[2:v]format=gbrp[e];[c][d][e]maskedmerge,format=yuv444p[s]")
    return fg, mask
