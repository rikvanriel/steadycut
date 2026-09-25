"""Single entry point: filename + interval in, stabilized clip out.

Runs the full pipeline: camera gate -> framing -> sync -> gyro path + render ->
residual certificate -> optional bounded 2D pass. Every auto-discovered value
carries its fallback; refusals say what to do instead of failing silently.

A camera body seen for the first time is MEASURED, not assumed: its gyro axis
map is derived from the user's own footage by steadycut.calibration.discover
and stored in the user's registry (~/.config/steadycut/cameras.json), so a
fresh install works on first use and the measurement survives upgrades.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stabilized, framed clip from a 360 .insv file")
    ap.add_argument("source", help=".insv file")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--out", default=None, help="output mp4 (default: auto)")
    ap.add_argument("--tall", action="store_true",
                    help="9:16 vertical (default 4:3 landscape)")
    ap.add_argument("--framing", default="mtb", metavar="POLICY",
                    help="framing policy for this mount (default mtb: camera "
                         "on the rider's helmet, bar in the bottom of frame). "
                         "A policy decides where the subject sits and how the "
                         "pitch is derived; the camera profile decides what "
                         "the lens can do")
    ap.add_argument("--pitch", type=float, default=None,
                    help="manual framing pitch (skips the policy's measurement)")
    ap.add_argument("--no-2d", action="store_true",
                    help="skip the bounded 2D residual pass")
    ap.add_argument("--preset", default="medium",
                    help="x264 preset for the final encode")
    ap.add_argument("--camera", default=None, metavar="MODEL",
                    help="force the camera model when the container carries "
                         "no recognizable signature")
    ap.add_argument("--corner", type=float, default=0.15,
                    help="gyro correction corner frequency in Hz (default "
                         "0.15): motion faster than this is removed; lower is "
                         "steadier but lags the rider's look direction. The "
                         "default is measured: 0.15 beat 0.5 on every window "
                         "of a multi-clip validation, and 0.05 bought nothing "
                         "while costing the near field")
    ap.add_argument("--image-only", action="store_true",
                    help="bounded 2D image stabilisation without telemetry: "
                         "no reframe, no gyro (for cameras with no embedded "
                         "IMU or pre-stabilised footage)")
    return ap


def refuse(msg: str, hint: str) -> int:
    print(f"REFUSE| {msg}\nREFUSE| {hint}")
    return 2


def measure_camera(source, start, ident, hint, tmp) -> dict:
    """Measure an unknown body's axis map, store it, and re-identify.

    The user's own footage is the only instrument that can supply a body's
    axis map, and a fresh install has none -- so this runs automatically the
    first time a camera is used instead of pointing the user at scripts. The
    result goes into the user's registry, so it happens once per body, and a
    failure says which windows were tried and what to do about it.
    """
    from steadycut.calibration.discover import discover_axis_map
    from steadycut.ingest.cameras.registry import identify, record_measured

    key = ident["measure_key"]
    print(f"CLI| {ident['model']} ({key}): {ident['reason']}")
    print("CLI| measuring this body's gyro axes from the file's own footage")
    got = discover_axis_map(source, start, workdir=tmp / "discover")
    for attempt in got.get("attempts", []):
        if "skipped" in attempt:
            print(f"CLI|   window {attempt['start']:.0f}s skipped: "
                  f"{attempt['skipped']}")
        else:
            print(f"CLI|   window {attempt['start']:.0f}s weakest |r| "
                  f"{attempt['strength']:.2f}")
    if not got["ok"]:
        return {"status": "refused",
                "reason": f"could not measure {ident['model']}: {got['reason']}",
                "action": got["action"]}
    axis_map = got["axis_map"]
    print("CLI| measured axis map " + "  ".join(
        f"{ch}=axis{col}{sign:+d}" for ch, (col, sign) in axis_map.items()))
    print(f"CLI|   stored in the camera registry")
    record_measured(key, model=ident.get("model"), fw=ident.get("fw"),
                    axis_map=axis_map,
                    measured={"source": str(source), "start": got["start"],
                              "fps": got["fps"], "frames": got["frames"],
                              "corr": got["corr"]},
                    note="measured from this body's own footage on first use")
    return identify(source, hint=hint)


def main(argv=None) -> int:
    from steadycut.core.pipeline import (
        ClipSpec, build_gyro_path, correlate, far_field_score,
        gyro_pitch_per_frame, hybrid_correct, measure_sync, plate_scale,
        render_constant, render_path, stabilize_image_only, track_far)
    from steadycut.framing import ladder
    from steadycut.framing.policies import policy_for, policy_names
    from steadycut.ingest.cameras.registry import identify

    a = build_parser().parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="steadycut-cli-"))
    cert: dict = {"source": str(a.source), "start": a.start, "dur": a.dur}

    # 1. camera gate: identify, then load the profile. A body with no stored
    #    measurement is measured HERE from the user's own footage and the
    #    result is stored in the user's registry, so first use is automatic.
    #    Refusals name the next step; a profile is never guessed.
    ident = identify(a.source, hint=a.camera)
    if ident["status"] == "needs_measure":
        ident = measure_camera(a.source, a.start, ident, a.camera, tmp)
    if ident["status"] != "ok":
        return refuse(ident["reason"], ident["action"])
    profile = ident["profile"]
    cert["camera"] = {"via": ident["via"], "model": ident["model"],
                      "serial": ident.get("serial"),
                      "capabilities": sorted(profile.capabilities)}
    if profile.axis_map is not None:
        cert["camera"]["axis_map"] = {ch: list(v) for ch, v in
                                      profile.axis_map.__dict__.items()}
    print(f"CLI| camera {ident['model']} via {ident['via']} "
          f"({'+'.join(sorted(profile.capabilities)) or 'no capabilities'})")

    # 1b. image-only mode: the bounded 2D pass IS the stabiliser. Used for
    #     cameras with no embedded IMU and for pre-stabilised footage.
    if a.image_only:
        out = (Path(a.out) if a.out
               else Path(f"clip_{int(a.start)}s_imageonly.mp4"))
        try:
            res = stabilize_image_only(a.source, out, a.start, a.dur,
                                       preset=a.preset)
        except Exception as exc:
            return refuse(f"image-only pass failed ({exc})", "no clip claimed")
        cert["image_only"] = {"raw_bounce": res["raw_bounce"],
                              "corrected_bounce": res["corrected_bounce"],
                              "inliers_med": res["inliers_med"]}
        print(f"CLI| image-only far bounce raw {res['raw_bounce']:.2f} -> "
              f"corrected {res['corrected_bounce']:.2f}")
        if res["corrected_bounce"] >= res["raw_bounce"]:
            print("CLI| WARN the 2D pass did not beat raw on this window")
        cert_path = out.with_suffix(".cert.json")
        cert_path.write_text(json.dumps(cert, indent=1, default=str))
        print(f"CLI| certificate {cert_path}")
        return 0

    # 1c. capability gates for the gyro/reframe path.
    if "gyro_stabilize" not in profile.capabilities:
        return refuse(f"{profile.model} carries no usable embedded IMU here",
                      "use --image-only (bounded 2D stabilisation), or supply "
                      "external telemetry (plan milestone E)")
    if profile.prestabilized_default:
        return refuse(f"{profile.model} bakes stabilisation into the file by "
                      "default",
                      "re-record with in-camera stabilisation OFF, or use "
                      "--image-only")
    if profile.axis_map is None:
        return refuse(f"{profile.model} has no measured axis map for this body",
                      "the measurement is automatic: re-run on a section where "
                      "the camera is moving and shaking, so the axes can be "
                      "correlated against image motion")

    fov = profile.max_useful_hfov["9:16" if a.tall else "4:3"]
    size = (540, 960) if a.tall else (960, 720)

    # 2. framing, by the policy for this mount. The policy owns the criterion
    #    and the way the pitch is derived; --pitch overrides both.
    policy = policy_for(a.framing)
    if policy is None:
        return refuse(f"no framing policy named {a.framing!r}",
                      f"known policies: {', '.join(policy_names())}")
    if a.pitch is not None:
        pitch = a.pitch
        cert["framing"] = {"pitch": pitch, "manual": True,
                           "policy": policy.name, "mount": policy.mount}
    elif policy.search is not None:
        try:
            pitch, info = ladder.solve(a.source, a.start, policy.search,
                                       workdir=tmp / "framing")
        except Exception as exc:
            return refuse(f"{policy.name} framing failed ({exc})", policy.hint)
        if not info.get("usable", False):
            return refuse(info.get("reason")
                          or f"{policy.name} could not measure a framing pitch",
                          policy.hint)
        cert["framing"] = {"pitch": pitch, "manual": False,
                           "policy": policy.name, "mount": policy.mount, **info}
    elif policy.pitch is not None:
        pitch = policy.pitch
        cert["framing"] = {"pitch": pitch, "manual": False,
                           "policy": policy.name, "mount": policy.mount,
                           "note": "policy fixed pitch"}
    else:
        return refuse(f"{policy.name} has neither a pitch nor a way to "
                      "measure one", policy.hint)
    print(f"CLI| framing pitch {pitch:.1f} "
          f"(policy {policy.name}, mount {policy.mount}, "
          f"follows {policy.follows})")
    spec = ClipSpec(source=a.source, start=a.start, duration=a.dur,
                    framing_pitch=pitch, fov=fov, size=size, profile=profile)

    # 3. sync (same window; validated by residual later, not by a 2nd window).
    sync = measure_sync(spec, workdir=tmp / "sync")
    if abs(sync.r_best) < 0.5:
        print(f"CLI| WARN weak sync r={sync.r_best:+.2f}: zero offset")
        offset_ms, sync_used = 0.0, False
    else:
        offset_ms, sync_used = -sync.lag_ms, True
    cert["sync"] = {"lag_ms": sync.lag_ms, "r": sync.r_best,
                    "offset_applied": offset_ms, "used": sync_used}
    print(f"CLI| sync lag {sync.lag_ms:.0f}ms r={sync.r_best:+.2f} "
          f"-> offset {offset_ms:+.0f}ms")

    # 4. gyro path + render.
    out = Path(a.out) if a.out else Path(f"clip_{int(a.start)}s.mp4")
    try:
        pts = build_gyro_path(spec, stamp_offset_ms=offset_ms,
                              corner_hz=a.corner)
        render_path(spec, pts, out, preset=a.preset)
    except Exception as exc:
        return refuse(f"render failed ({exc})", "no clip claimed")
    print(f"CLI| wrote {out}")

    # 5. bounded 2D residual pass -- BEFORE the certificate, so the numbers
    #    describe the file that is actually delivered rather than the gyro
    #    stage underneath it. Its own track supplies the inlier count the
    #    weak-tracking gate needs.
    if not a.no_2d:
        staged = tmp / "hybrid.mp4"
        pre = hybrid_correct(out, staged)
        if float(np.median(pre.inliers)) < 150:
            cert["two_d"] = {"applied": False, "reason": "weak tracking"}
            print("CLI| WARN weak tracking: skipping 2D pass, gyro clip stands")
        else:
            # Never warp in place: write aside, then atomically replace.
            staged.replace(out)
            # Record what was actually run, read from the production defaults
            # rather than restated: the certificate describes the delivered
            # file, and that includes the pass that produced it.
            import inspect
            hy = inspect.signature(hybrid_correct).parameters
            cert["two_d"] = {"applied": True,
                             "sigma": hy["sigma"].default,
                             "bound": hy["bound"].default,
                             "zoom": hy["zoom"].default}
            print("CLI| 2D residual pass applied")
    else:
        cert["two_d"] = {"applied": False, "reason": "--no-2d"}

    # 6. residual certificate, measured on the delivered file with the
    #    validated estimator: the far field is the landscape, and it carries
    #    TWO numbers because sway and jitter are different faults.
    traj = track_far(out)
    dy = traj.deltas[:, 1] / plate_scale(spec.fov)[1]
    pr = gyro_pitch_per_frame(a.source, a.start, len(dy),
                              axis_map=profile.axis_map)
    r = correlate(pr, dy)
    raw = tmp / "raw.mp4"
    render_constant(spec, raw)
    corrected_score = far_field_score(out, render_size=size)
    raw_score = far_field_score(raw, render_size=size)
    cert["residual_r"] = r
    cert["far_field"] = {"render_size": list(size),
                         "corrected": corrected_score, "raw": raw_score}
    print(f"CLI| residual-vs-gyro r={r:+.3f} (small = explained part removed)")
    print(f"CLI| far field rms {raw_score['rms_px']:.2f} -> "
          f"{corrected_score['rms_px']:.2f} px  (jitter "
          f"{raw_score['jitter_px']:.2f} -> {corrected_score['jitter_px']:.2f})")
    if abs(r) > 0.5:
        print("CLI| WARN correction unverified: residual still gyro-locked")

    cert["corner_hz"] = a.corner
    cert_path = out.with_suffix(".cert.json")
    cert_path.write_text(json.dumps(cert, indent=1, default=str))
    print(f"CLI| certificate {cert_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
