"""Factory geometry from the file's own makernotes.

telemetry_parser decodes the Metadata block: serial, fw, per-lens circle
fit + half-FOV + rotation quaternion (offset_v2/v3), rolling_shutter_time,
gyro cal. This module turns that into a plain dict the fit (Step 2) seeds
from — factory values, never our served fitted numbers.
"""
from __future__ import annotations


def factory_geometry(path: str) -> dict:
    """Per-file factory geometry. Raises RuntimeError when unavailable."""
    import telemetry_parser as tp
    meta = tp.telemetry_parser.Parser(path).telemetry(
        human_readable=False)[0]['Default']['Metadata']
    return {
        'serial': meta.get('serial_number', ''),
        'fw': meta.get('fw_version', ''),
        'rolling_shutter_s': float(meta['rolling_shutter_time']) / 1000.0,
        'offset_v2': [float(x) for x in meta['offset_v2']],
        'offset_v3': [float(x) for x in meta['offset_v3']],
        'pts_type': meta.get('pts_type'),
    }


def lens_circle(v2: list, lens: int) -> dict:
    """Circle center + radius + half-FOV for lens 0/1 from offset_v2.

    Layout per lens block: (cx, cy, r, a, b, half_fov, ...) — the
    image-circle fit (center x/y + radius) plus the half field of view.
    Blocks are 16 entries: base = 1 + lens * 16.
    """
    base = 1 + lens * 16
    cx, cy, r = v2[base], v2[base + 1], v2[base + 2]
    half_fov = v2[base + 5]
    return {'cx': cx, 'cy': cy, 'r': r, 'half_fov_deg': half_fov,
            'full_fov_deg': 2 * half_fov}


def fit_lens_circle(frame, thresh=8.0) -> dict:
    """Image-circle fit (center x/y + radius) on a fisheye frame.

    Boundary rays from frame center: first dark pixel past the rim;
    algebraic fit with iterative outlier rejection (dark content can
    stop a ray early). Returns cx, cy, r in frame pixels + inlier count.
    """
    import cv2
    import numpy as np
    gray = frame if np.asarray(frame).ndim == 2 else cv2.cvtColor(
        np.asarray(frame), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # small blur: just enough to kill sensor noise; a wide blur would
    # push the dark-edge outward by ~2 sigma (measured 11px at sigma 6).
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    pts = []
    for ang in np.linspace(0, 2 * np.pi, 1440, endpoint=False):
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        for r in range(w // 2 - 200, w // 2 + 200, 2):
            x, y = int(w / 2 + dx * r), int(h / 2 + dy * r)
            if 0 <= x < w and 0 <= y < h and float(blur[y, x]) < thresh:
                pts.append((x, y))
                break
    p = np.array(pts, float)
    if len(p) < 100:
        raise ValueError(f'circle fit: only {len(p)} boundary rays')
    for _ in range(5):
        A = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
        b = -(p[:, 0] ** 2 + p[:, 1] ** 2)
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy = -D / 2, -E / 2
        r = float(np.sqrt((D ** 2 + E ** 2) / 4 - F))
        resid = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        p = p[resid < 12]
    return {'cx': float(cx), 'cy': float(cy), 'r': r, 'n': int(len(p))}


def lens_quat(v2: list, lens: int) -> tuple:
    """Per-lens rotation offset (w, x, y, z) from offset_v2."""
    base = 1 + lens * 16
    return (v2[base + 9], v2[base + 10], v2[base + 11], v2[base + 12])
