"""Per-recording optics selection (Task 1.3): serial -> profile -> spec.

Scores stored profiles plus a fresh grid fit on textured frames; the winner
within tolerance serves. A fresh fit that lands on a stored candidate
CONFIRMS it (the second-recording rule, automatic); one that lands nowhere
enrolls a new namespaced candidate and serves per-file until confirmed.
"""
from __future__ import annotations

from steadycut.ingest.cameras.registry import (OPTICS_FOV_EPS,
                                                confirm_optics, optics_for,
                                                record_optics)
from steadycut.render.fov_fit import fit, render_pair, score_sphere

TOL_EDGE = 0.10  # winner must be within 10% edge-ratio of the fresh best


def _frames(source, times=(10.0, 30.0, 60.0)):
    import subprocess
    import tempfile
    cv2 = __import__("cv2")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            pair = []
            for trk in (0, 1):
                p = f"{tmp}/L{trk}.png"
                subprocess.run(
                    f"ffmpeg -hide_banner -v error -ss {t} -i '{source}' "
                    f"-map 0:v:{trk} -frames:v 1 -y {p}",
                    shell=True, check=True, timeout=180)
                pair.append(cv2.imread(p))
            if pair[0] is not None and pair[1] is not None:
                out.append(tuple(pair))
    return out


def _score_profile(pairs, fa, fb):
    scores = [score_sphere(render_pair(A, B, fa, fb)) for A, B in pairs]
    return sum(s[0] for s in scores) / len(scores)


def select_optics(source, serial, fw, serial_known):
    """Returns (profile_dict, fresh_fit_bool). profile_dict carries fov_a,
    fov_b, deflare_poly, gain_default, id (or None when refused)."""
    pairs = _frames(source)
    if not pairs:
        return {"id": None}, True
    stored = (optics_for(serial, fw) if serial_known else [])
    fresh = fit(*pairs[0])
    ffa, ffb, _, _ = fresh
    scored = [(p, _score_profile(pairs, p["fov_a"], p["fov_b"]))
              for p in stored
              if "fov_a" in p and "fov_b" in p]
    scored.sort(key=lambda r: r[1])
    fresh_score = _score_profile(pairs, ffa, ffb)
    if scored and scored[0][1] <= fresh_score * (1 + TOL_EDGE):
        win = dict(scored[0][0])
        if win.get("status") == "candidate" and serial_known:
            # Serving a candidate on a second recording IS the confirmation.
            confirm_optics(serial, win["id"], str(source))
            win["status"] = "confirmed"
        return win, False
    if not serial_known:
        return {"id": None, "fov_a": ffa, "fov_b": ffb}, True
    # fresh fit: confirm a nearby candidate or enroll a new one.
    for p in stored:
        try:
            if (abs(float(p["fov_a"]) - ffa) <= OPTICS_FOV_EPS
                    and abs(float(p["fov_b"]) - ffb) <= OPTICS_FOV_EPS):
                confirm_optics(serial, p["id"], str(source))
                return dict(p, id=p["id"]), True
        except (KeyError, TypeError, ValueError):
            continue
    entry = record_optics(serial, {"fov_a": ffa, "fov_b": ffb,
                                   "fw": fw}, serial_known=True)
    new = entry["optics"][-1]
    return dict(new), True
