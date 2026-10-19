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
    M = repair_spans(A, B, wgt, [span], reroute=False, multiband=False)
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
    # passthrough: caller keeps its own (multiband) img.
    assert M is None


def _local_double(dx=60, seed=13):
    """Copied bar on independent backgrounds (zero-offset never matches,
    the displaced copy matches exactly -- the washed-snow regime)."""
    rng = np.random.default_rng(seed)
    A = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B[100:160, 1514:1594] = A[100:160, 1470:1550]
    return A, B


def test_auto_repair_fires_on_shifted_pair():
    # T1 FREEZE (serial-geometry fit plan): 44px auto-warps suspend
    # until the correspondence fit lands — with current geometry the
    # excess cannot be told from a two-person pair. Asserts refusal
    # by name; revisit after the fit (budget will admit true doubles).
    from steadycut.render.nearfield import auto_repair
    A, B = _local_double()
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 1536:] = 1.0
    M, cert = auto_repair(A, B, wgt)
    assert M is None
    assert cert["spans"], "detection must still fire"
    assert all(s["gate_refused"] == "dx-exceeds-parallax-budget"
               for s in cert["spans"]), cert["spans"]



def test_reroute_flag_runs_and_keeps_single():
    from steadycut.render.nearfield import repair_spans
    A, B = _shifted(dx=40)
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 1536:] = 1.0
    span = ((100, 160, 1460, 1545), 40.0, 1536)
    M = repair_spans(A, B, wgt, [span], reroute=True, multiband=True)
    plain = (A.astype(float) * (1 - wgt[:, :, None])
             + B.astype(float) * wgt[:, :, None])
    zone = (slice(100, 160), slice(1500, 1600))
    assert (np.abs(M[zone].astype(float) - A[zone].astype(float)).mean()
            < np.abs(plain[zone].astype(float) - A[zone].astype(float)
                     ).mean())
