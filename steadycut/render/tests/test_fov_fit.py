"""Pins for the production FOV fit (Task 1.2): metric behavior + refinement.

Scope honesty (review catch): synthetic spheres from build_maps would test
the builder against itself (circular), so unit tests here cover the SCORE
(an independent synthetic strip: aligned must beat shifted) and the REFINE
convergence on a convex curve. Real proof is the transfer test — fit on clip
A, optimum transfers to clip B same body — which needs footage and runs as
an instrument (Task 1.4), not here.
"""
import numpy as np

from steadycut.render.fov_fit import _refine, score_sphere


def _strip(shifted=False):
    """Synthetic equirect: content either continues across the x=256 seam
    (aligned) or cuts at it (misaligned: bar ends, different plateau starts
    — the edge a misregistered seam leaves behind)."""
    img = np.full((512, 1024, 3), 128, np.uint8)
    img[:, 100 - 4:100 + 4] = 200  # baseline texture elsewhere
    if not shifted:
        img[:, 256 - 4:256 + 4] = 200  # bar continues through the seam
    else:
        img[:, 248:256] = 200
        img[:, 256:] = np.where(img[:, 256:] == 128, 60, img[:, 256:])
    x1 = 768
    img[:, x1 - 4:x1 + 4] = 200
    if shifted:
        img[:, x1:x1 + 8] = 60
    return img


def test_score_prefers_aligned_over_shifted():
    r_ok, _ = score_sphere(_strip(False))
    r_bad, _ = score_sphere(_strip(True))
    assert r_ok < r_bad, (r_ok, r_bad)


def test_score_step_reads_plateau_difference():
    img = np.full((512, 1024, 3), 100, np.uint8)
    img[:, 256:] = 130  # step at the x=256 seam column
    _, step = score_sphere(img)
    assert abs(step - 90.0) < 5.0  # 3 channels x 30 DN


def test_refine_converges_on_convex_curve():
    import steadycut.render.fov_fit as F

    def fake_render(A, B, fa, fb, w=1024, h=512):
        return (fa, fb)  # pass-through: score reads the params

    orig_render, orig_score = F.render_pair, F.score_sphere
    try:
        F.render_pair = fake_render
        F.score_sphere = lambda pair: ((pair[0] - 183.5) ** 2
                                       + (pair[1] - 181.0) ** 2, 0.0)
        fa, fb, _ = F._refine(None, None, 182.0, 180.0)
        assert abs(fa - 183.5) <= 0.25 and abs(fb - 181.0) <= 0.25, (fa, fb)
    finally:
        F.render_pair, F.score_sphere = orig_render, orig_score
