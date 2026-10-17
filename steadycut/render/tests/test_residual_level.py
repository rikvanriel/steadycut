"""residual_level: DC step removal on flat, refusal on content."""
import numpy as np

from steadycut.render.nearfield import residual_level


def _halves(step_dn, texture_row=None, bar_diff=0.0):
    h, w = 200, 400
    A = np.full((h, w, 3), 180.0, np.float32)
    B = np.full((h, w, 3), 180.0 + step_dn, np.float32)
    if texture_row is not None:
        # vertical bar crossing the seam: same tone step underneath, but
        # bar_diff extra content difference on the B side (parallax/occluder)
        A[texture_row[0]:texture_row[1], 150:250] = 40.0
        B[texture_row[0]:texture_row[1], 150:250] = 40.0 + bar_diff
    wgt = np.clip((np.arange(w)[None, :].repeat(h, axis=0) - 150) / 100,
                  0, 1)
    return A, B, wgt


def test_flat_step_removed():
    A, B, wgt = _halves(6.0)
    A2, B2, _ = residual_level(A, B, wgt, feather_frac=0.375)
    # at the seam both meet at 183 (seam column is A-side, next is B-side)
    assert abs(float(A2[:, 200].mean()) - 183.0) < 1.0
    assert abs(float(B2[:, 201].mean()) - 183.0) < 1.0
    # far field untouched
    assert abs(float(A2[:, 0].mean()) - 180.0) < 0.5
    assert abs(float(B2[:, -1].mean()) - 186.0) < 0.5


def test_content_edge_refused():
    # bar 40 DN darker on A only: row step 40 > max_dn -> refuse, unpainted
    A, B, wgt = _halves(6.0, texture_row=(80, 120), bar_diff=40.0)
    A2, B2, _ = residual_level(A, B, wgt, feather_frac=0.375)
    assert abs(float((B2[100, 201] - A2[100, 200]).mean()) - 40.0) < 2.0
    # flat rows still equalized
    assert abs(float((B2[10, 201] - A2[10, 200]).mean())) < 1.0


def test_big_step_refused():
    A, B, wgt = _halves(30.0)  # real edge, not tone
    A2, B2, _ = residual_level(A, B, wgt, feather_frac=0.375)
    assert abs(float((B2[:, 201] - A2[:, 200]).mean()) - 30.0) < 1.0


def test_crease_removed():
    # same level at the seam, opposite slopes (vignetting crease):
    # A falls toward the seam, B rises away from it
    h, w = 200, 400
    x = np.arange(w, dtype=np.float32)
    A = (180.0 - 0.05 * (x - 200)).astype(np.float32)
    B = (180.0 + 0.05 * (x - 200)).astype(np.float32)
    A = np.broadcast_to(A[None, :], (h, w)).copy()
    B = np.broadcast_to(B[None, :], (h, w)).copy()
    A = A[:, :, None].repeat(3, axis=2)
    B = B[:, :, None].repeat(3, axis=2)
    wgt = np.clip((np.arange(w)[None, :].repeat(h, axis=0) - 150) / 100,
                  0, 1)
    A2, B2, _ = residual_level(A, B, wgt, feather_frac=0.375)
    # slopes each side of the seam now agree (C1: kink gone); measure
    # close in (the feather relaxes further out, which is the point)
    a = A2[:, :, 0].mean(axis=0)
    b = B2[:, :, 0].mean(axis=0)
    xx = np.arange(w, dtype=float)
    sL = np.polyfit(xx[180:200], a[180:200], 1)[0]
    sR = np.polyfit(xx[200:220], b[200:220], 1)[0]
    assert abs(sR - sL) < 0.02, (sL, sR)


def test_rim_ring_removed():
    # one-lens rim ring (in-camera vignetting overshoot): localized bump
    # on A near the seam, B smooth — the quadratic basis catches it
    h, w = 200, 400
    x = np.arange(w, dtype=float)
    A = np.full((h, w), 180.0) + 4.0 * np.exp(-((x - 190) / 8) ** 2)[None, :]
    B = np.full((h, w), 180.0)
    A = np.ascontiguousarray(A[:, :, None].repeat(3, axis=2), np.float32)
    B = np.ascontiguousarray(B[:, :, None].repeat(3, axis=2), np.float32)
    wgt = np.clip((np.arange(w)[None, :].repeat(h, axis=0) - 150) / 100,
                  0, 1)
    # production-like scales: scene smooth much wider than the bump, so S
    # does not absorb what the correction must remove
    A2, B2, _ = residual_level(A, B, wgt, feather_frac=0.375,
                               sigma_scene_frac=0.15)
    got = float(A2[:, 185:195, 0].mean() - 180.0)
    assert abs(got) < 1.5, got


def test_chroma_flatten_gray_only():
    # gray pixels near the seam lose their tint; saturated keep color
    h, w = 200, 400
    A = np.full((h, w, 3), 180.0, np.float32)
    B = np.full((h, w, 3), 180.0, np.float32)
    A[:, :, 0] += 6.0  # blue tint one side
    B[:, :, 2] += 6.0  # red tint other side
    A[150:, :, 0] = 60.0  # saturated red band (jacket): untouched
    B[150:, :, 0] = 60.0
    wgt = np.clip((np.arange(w)[None, :].repeat(h, axis=0) - 150) / 100,
                  0, 1)
    A2, B2, _ = residual_level(A, B, wgt, feather_frac=0.375)
    # gray zone tint reduced at the seam
    assert abs(float(A2[10, 200, 0] - A2[10, 200, 2])) < 3.0
    assert abs(float(B2[10, 201, 0] - B2[10, 201, 2])) < 3.0
    # red band keeps its saturation
    assert float(A2[160, 200, 2] - A2[160, 200, 0]) > 100.0


def test_edge_flatten_one_sided_ramp():
    from steadycut.render.nearfield import edge_flatten
    # one lens carries a 20 DN red ramp at its edge; the other is clean.
    # Common-mean leveling cannot see it; per-lens flatten removes it.
    h, w = 200, 400
    x = np.arange(w, dtype=float)
    ramp = 20.0 * np.clip((x - 150) / 50, 0, 1) * (x < 200)
    A = np.full((h, w, 3), 180.0, np.float32)
    A[:, :, 2] += ramp[None, :].astype(np.float32)
    B = np.full((h, w, 3), 180.0, np.float32)
    wgt = np.clip((np.arange(w)[None, :].repeat(h, axis=0) - 150) / 100,
                  0, 1)
    A2, B2, info = edge_flatten(A, B, wgt, feather_frac=0.375)
    got = float((A2[:, 175:195, 2] - 180.0).mean())
    assert abs(got) < 5.0, got
    # clean lens untouched
    assert abs(float((B2 - 180.0).max())) < 1.0
