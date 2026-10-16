"""Pins for near-field repair: displace, repair, auto (synthetic)."""
import numpy as np


def _shifted(h=256, w=2048, dx=40, seed=11):
    rng = np.random.default_rng(seed)
    A = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    B = np.zeros_like(A)
    B[:, dx:] = A[:, :w - dx]  # B-side double: same content dx right
    return A, B


def test_span_displacement_recovers_shift():
    from steadycut.render.nearfield import span_displacement
    A, B = _shifted()
    dx, n = span_displacement(A, B, (60, 200, 1500, 1620))
    assert dx is not None and abs(dx - 40) <= 4, (dx, n)


def test_span_displacement_refuses_bland():
    from steadycut.render.nearfield import span_displacement
    A = np.full((256, 2048, 3), 180, np.uint8)
    dx, n = span_displacement(A, A.copy(), (60, 200, 1500, 1620))
    assert dx is None


def test_repair_spans_collapses_double():
    from steadycut.render.nearfield import repair_spans
    A, B = _shifted()
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 1536:] = 1.0  # hard cut at the seam: B shown right
    span = ((100, 160, 1460, 1545), 40.0, 1536)
    M = repair_spans(A, B, wgt, [span])
    ref = A[100:160, 1460:1545].astype(float)
    got = M[100:160, 1460:1545].astype(float)
    assert np.abs(got - ref).mean() < 12.0


def test_auto_repair_clean_pair_is_plain_merge():
    from steadycut.render.nearfield import auto_repair
    A, B = _shifted(dx=0)
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 1536:] = 1.0
    M, cert = auto_repair(A, B, wgt)
    assert cert["spans"] == []
    ref = (A.astype(float) * (1 - wgt[:, :, None])
           + B.astype(float) * wgt[:, :, None])
    assert np.abs(M.astype(float) - ref).mean() < 1.0


def test_auto_repair_fires_on_shifted_pair():
    from steadycut.render.nearfield import auto_repair
    A, B = _shifted(dx=40)
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 1536:] = 1.0
    M, cert = auto_repair(A, B, wgt)
    fired = [s for s in cert["spans"] if s["action"] == "warp+trim+patch"]
    assert fired, [s["action"] for s in cert["spans"]]
    plain = (A.astype(float) * (1 - wgt[:, :, None])
             + B.astype(float) * wgt[:, :, None])
    zone = (slice(60, 200), slice(1450, 1650))
    assert (np.abs(M[zone].astype(float) - A[zone].astype(float)).mean()
            < np.abs(plain[zone].astype(float) - A[zone].astype(float)
                     ).mean())


def test_lens_validity_axes_and_rims():
    from steadycut.render.nearfield import lens_validity
    vA, vB = lens_validity(1024, 2048)
    assert vA[512, 1024] and not vB[512, 1024]  # A axis: A only
    assert vB[512, 0] and not vA[512, 0]        # B axis: B only
    assert vA[512, 1536] and vB[512, 1536]      # seam: both (90 deg)
    assert not vA[512, 1620]                    # past A model: masked


def test_past_model_pixels_fall_to_other_lens():
    from steadycut.render.nearfield import repair_spans
    H, W = 256, 2048
    A = np.full((H, W, 3), 180, np.uint8)
    B = np.full((H, W, 3), 60, np.uint8)
    wgt = np.zeros((H, W), np.float32)  # would show A everywhere left
    M = repair_spans(A, B, wgt, [((100, 200, 1600, 1650), None, 1536)],
                     fov_a=185.0, fov_b=183.5)
    assert M[150, 1620, 0] < 100  # A past-model there: B shown, not A
