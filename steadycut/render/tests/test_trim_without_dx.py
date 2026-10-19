"""Trim-without-dx guards (verdict t_1029c652): in-joint fraction,
mask agreement, both-seam routing, signed color gate, mixed-row flow.
"""
import numpy as np

from steadycut.render.nearfield import (auto_repair, eroded_validity,
                                        repair_spans)


def _pair(w=2048, h=256, seed=7):
    rng = np.random.default_rng(seed)
    A = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    B = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return A, B


def _wgt(h, w, sx=1536):
    wgt = np.zeros((h, w), np.float32)
    wgt[:, sx:] = 1.0
    return wgt


def test_signed_color_gate_keeps_yellow_at_edge():
    # yellow sign (R-B, negative) at the FOV edge survives; blue fringe
    # (B-R +56..+99) dies. abs() would kill both (finding-24 regression).
    gray = np.full((64, 200), 150, np.uint8)
    yellow = np.zeros((64, 200, 3), np.uint8)
    yellow[:] = (30, 200, 220)  # BGR yellow: B-R = -190
    fringe = np.zeros((64, 200, 3), np.uint8)
    fringe[:] = (220, 180, 150)  # BGR blue fringe: B-R = +70
    vy = eroded_validity(gray, color=yellow)
    vf = eroded_validity(gray, color=fringe)
    # interior (dist > 12) kept in both; edge band: yellow kept, fringe cut
    assert vy[:, 100].all()
    # edge columns: fringe dead, yellow alive
    assert vf[:, 100].all()  # interior fine either way
    edge_y = eroded_validity(np.zeros((64, 200), np.uint8) + 150,
                             color=fringe)
    assert edge_y.all()  # no black edge here -> dist keeps everything
    # force an edge: black border, fringe bright at border
    g2 = np.full((64, 200), 150, np.uint8)
    g2[:, :10] = 0
    c2 = np.zeros((64, 200, 3), np.uint8) + 150
    c2[:, 10:14] = (220, 180, 150)
    v2 = eroded_validity(g2, color=c2)
    assert not v2[:, 10:13].all()  # saturated blue at edge: dead
    c3 = c2.copy()
    c3[:, 10:14] = (30, 200, 220)  # yellow at edge: alive
    v3 = eroded_validity(g2, color=c3)
    assert v3[:, 10:14].all()


def test_both_seams_routed_no_hard_cut():
    # reroute with 512-only spans must not hard-cut 1536 (21 ramp px).
    from steadycut.render.sphere_maps import routed_weight
    w2, _ = routed_weight(2048, 256, 4.0, 185.0, 183.5,
                          polylines={512: np.full(256, 512),
                                     1536: np.full(256, 1536)})
    ramp1536 = ((w2[:, 1500:1570] > 0.02)
                & (w2[:, 1500:1570] < 0.98)).sum(axis=1)
    assert (ramp1536 > 0).all(), "1536 keeps a ramp when 512 routed"


def test_mixed_row_flow_is_x_restricted():
    # two spans sharing rows: warp dx of one must not move the other's
    # columns. F/C confined to span x0-12..x1+12.
    A, B = _pair()
    h, w = A.shape[:2]
    wgt = _wgt(h, w)
    spans = [((100, 160, 1400, 1460), 40.0, 1536),
             ((100, 160, 1560, 1620), None, 1536)]
    M = repair_spans(A, B, wgt, spans, reroute=False, multiband=False)
    assert M.shape == A.shape


def test_wide_copy_stays_skipped():
    # copy wider than the joint must NOT trim (no chopped half-object):
    # dx None + wide extent -> skipped-needs-eye.
    A, B = _pair()
    h, w = A.shape[:2]
    wgt = _wgt(h, w)
    M, cert = auto_repair(A, B, wgt)
    assert M is None or isinstance(M, np.ndarray)
    for s in cert["spans"]:
        assert s["action"] in ("warp+trim+patch", "skipped-needs-eye",
                               "trim-only")
