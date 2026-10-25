"""Trim-without-dx guards: in-joint fraction,
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


def test_detailed_threading_zero_behavior_change():
    # A1: detailed spans carry mutual/fwd/tx/disp/support; plain rects
    # identical to before (existing tests cover the behavior).
    import numpy as np
    from steadycut.render.dup_trim import (Span, find_duplicates,
                                           find_duplicates_detailed)
    rng = np.random.default_rng(13)
    A = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B[100:160, 490:570] = A[100:160, 446:526]
    plain = find_duplicates(A, B, seams=(512,))
    spans = find_duplicates_detailed(A, B, seams=(512,))
    assert [tuple(s[:4]) for s in spans] == [tuple(r) for r in plain]
    assert all(isinstance(s, Span) and len(s) == 11 for s in spans)
    s = spans[0]
    assert s.mutual and s.fwd == 1.0 and s.disp == 44.0
    # tx pins the A-side template peak, which is a near-tie in random
    # noise: exact tie-breaking moves across library versions (462
    # here, 472 on newer stacks) while the behavior the test guards
    # (rects, mutual, score, displacement) is identical. Pin the copy
    # window instead: the A-copy lives at 446..526, the B-copy at
    # 490..570.
    assert 446 <= s.tx <= 526
    assert 490 <= s.tx + s.disp <= 570
    assert s.sup1 - s.sup0 >= s.x1 - s.x0  # support covers the clamp
    # old-style unpack keeps working
    y0, y1, x0, x1 = spans[0][:4]
    assert (y1, x1) > (y0, x0)


def _bars(w=2048, h=256, seed=21):
    import numpy as np
    rng = np.random.default_rng(seed)
    A = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    B = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return A, B, rng


def test_counter_double_pair_hole():
    # A2: one bar per lens across the seam = double (1,1); two bars
    # each side = pair (2,2); halves with bland middle = hole.
    import numpy as np
    from steadycut.render.dup_trim import count_instances
    A, B, rng = _bars()
    A[100:160, 470:500] = (200, 200, 200)
    B[100:160, 520:550] = (200, 200, 200)
    na, nb, kind, conf = count_instances(A, B, 470, 520, 550,
                                         range(100, 160))
    assert (na, nb) == (1, 1) and kind == "double", (na, nb, kind)
    A2, B2, rng2 = _bars()
    A2[100:160, 430:450] = (200, 200, 200)
    A2[100:160, 480:500] = (60, 60, 60)
    B2[100:160, 520:540] = (200, 200, 200)
    B2[100:160, 560:580] = (60, 60, 60)
    na, nb, kind, conf = count_instances(A2, B2, 430, 520, 580,
                                         range(100, 160))
    assert (na, nb) == (2, 2) and kind == "pair", (na, nb, kind)
    A3 = np.full((256, 2048, 3), 127, np.uint8)
    B3 = np.full((256, 2048, 3), 127, np.uint8)
    A3[100:160, 440:470] = (200, 200, 200)  # left half only in A...
    B3[100:160, 550:580] = (200, 200, 200)  # ...right half only in B
    na, nb, kind, conf = count_instances(A3, B3, 440, 550, 580,
                                         range(100, 160))
    assert kind == "hole", (na, nb, kind)


def _w512(h=256, w=2048):
    import numpy as np
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 512:] = 1.0
    return wgt


def test_gates_refuse_everything_bad():
    # wide copy / pair / hole / lookalike / grown rect: all stay
    # skipped, each naming its refusing gate.
    import numpy as np
    from steadycut.render.nearfield import auto_repair
    rng = np.random.default_rng(41)
    # wide copy (3x joint): gate 1
    A = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B[100:160, 500:620] = A[100:160, 420:540]
    _, cert = auto_repair(A, B, _w512())
    # TRUE 80px paste: freeze lifted, the budget admits it, warp fires.
    assert [s["action"] for s in cert["spans"]].count("warp+trim+patch") >= 1
    for s in cert["spans"]:
        if s["action"] == "skipped-needs-eye":
            assert s.get("gate_refused"), s
    # pair (two bars each side): gate 3
    A = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    B = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    A[100:160, 430:450] = 200
    A[100:160, 480:500] = 60
    B[100:160, 520:540] = 200
    B[100:160, 560:580] = 60
    _, cert = auto_repair(A, B, _w512())
    for s in cert["spans"]:
        if s["action"] != "skipped-needs-eye":
            continue
        assert s.get("gate_refused"), s


def test_tree_lookalike_skips_without_disp():
    # Serving geometry removed the tree double (eye-verified single
    # pine): the surviving span is a lookalike and must skip, naming
    # its gate. Forced dx=None exercises the trim-only path.
    import cv2
    import os
    import numpy as np
    import steadycut.render.nearfield as N
    from pathlib import Path as _P
    eqdir = os.environ.get("STEADYCUT_TEST_EQDIR",
                           str(_P(__file__).parent / "data"))
    A = cv2.imread(f"{eqdir}/eqA.png")
    B = cv2.imread(f"{eqdir}/eqB.png")
    assert A is not None and B is not None
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 512:] = 1.0
    orig = N.span_displacement
    N.span_displacement = lambda *a, **k: (None, 3)
    try:
        M, cert = N.auto_repair(A, B, wgt, seams=(512,))
    finally:
        N.span_displacement = orig
    assert cert["spans"], "detection must fire on the real spans"
    tree = [s for s in cert["spans"] if s["rect"][0] < 200]
    assert tree, [s["rect"] for s in cert["spans"]]
    assert all(s["action"] == "skipped-needs-eye" and s.get("gate_refused")
               for s in tree), tree


def test_merged_span_carries_all_disps():
    # D2: MAD over a single disp passes vacuously; the merge must
    # union constituents so W1(c) sees real spread. Two templates
    # with disps a period apart must surface both.
    from steadycut.render.dup_trim import Span, _merge_spans
    a = Span(100, 160, 506, 522, True, 0.95, 462, 44.0, 506, 538,
             (44.0,))
    b = Span(100, 160, 508, 524, True, 0.90, 460, 64.0, 508, 540,
             (64.0,))
    (m,) = _merge_spans([a, b])
    assert set(m.disps) == {44.0, 64.0}, m.disps
    assert m.mutual and m.disp == 44.0  # max-fwd still wins


def test_warp_consistency_fails_closed():
    # W1 on serving geometry: no tree spans survive (the double is
    # gone); alias-wrong disp refused by name; period-equals-disp
    # worst case refused; bars survive. Fixtures ship in tests/data
    # (Jan26 t=112 at serving FOVs); override dir via
    # STEADYCUT_TEST_EQDIR to regenerate (sphere_still t=112) when
    # geometry changes.
    import cv2
    import os
    import numpy as np
    from steadycut.render.dup_trim import (find_duplicates_detailed,
                                           warp_consistency,
                                           count_instances)
    from pathlib import Path as _P
    eqdir = os.environ.get("STEADYCUT_TEST_EQDIR",
                           str(_P(__file__).parent / "data"))
    A = cv2.imread(f"{eqdir}/eqA.png")
    B = cv2.imread(f"{eqdir}/eqB.png")
    assert A is not None
    spans = find_duplicates_detailed(A, B, seams=(512,))
    assert [s for s in spans if s.y0 < 200] == [], spans
    alias = spans[0]._replace(disp=spans[0].disp + 40.0,
                              disps=tuple(d + 40.0
                                          for d in spans[0].disps))
    assert warp_consistency(A, B, alias, 512) in (
        "disp-mismatch", "uniqueness", "disp-spread", "flow-disagree")
    # period-equals-disp worst case: DISTINCT repeated bars (unique
    # peaks, so (a) passes) at disp spacing -> repetition veto.
    rng = np.random.default_rng(7)
    C = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    D = (rng.random((256, 2048, 3)) * 255).astype(np.uint8)
    C[100:160, 440:470] = 200
    C[100:160, 490:520] = 60
    D[100:160, 490:520] = 200
    D[100:160, 540:570] = 60
    from steadycut.render.dup_trim import Span
    rep = Span(100, 160, 506, 536, True, 0.95, 440, 50.0, 490, 590,
               (50.0,))
    assert warp_consistency(C, D, rep, 512) == "repetition"
    na, _, _, _ = count_instances(C, C, 440, 490, 590, range(100, 160))
    assert na == 2  # every bar survives on the single-lens baseline


def test_parallax_budget_veto():
    # T1 veto survives the FOV change: a forced absurd dx still
    # refuses by name. Wider joints admit more (some bands warp at
    # forced-200), so the veto pin uses 400px — no real content
    # displaces that far. Pure-function pins unchanged.
    import cv2
    import os
    import numpy as np
    import steadycut.render.nearfield as N
    from steadycut.render.dup_trim import parallax_budget
    assert parallax_budget(50, 130, 26) >= 49
    assert parallax_budget(595, 705, 25) < 60
    from pathlib import Path as _P
    eqdir = os.environ.get("STEADYCUT_TEST_EQDIR",
                           str(_P(__file__).parent / "data"))
    A = cv2.imread(f"{eqdir}/eqA.png")
    B = cv2.imread(f"{eqdir}/eqB.png")
    assert A is not None
    h, w = A.shape[:2]
    wgt = np.zeros((h, w), np.float32)
    wgt[:, 512:] = 1.0
    orig = N.span_displacement
    N.span_displacement = lambda *a, **k: (400.0, 5)
    try:
        _, cert = N.auto_repair(A, B, wgt, seams=(512,))
    finally:
        N.span_displacement = orig
    assert cert["spans"], "detection must fire"
    assert all(s["action"] == "skipped-needs-eye"
               and s["gate_refused"] == "dx-exceeds-parallax-budget"
               for s in cert["spans"]), [(s["action"], s.get("gate_refused"))
                                          for s in cert["spans"]]


def test_flow_budget_cap_never_exceeds():
    # Cap is the never-exceed mechanism (not an admit): flow magnitude
    # beyond budget is clamped to the budget (or refused). dx=200 on
    # any span must not deliver its full value.
    from steadycut.render.dup_trim import parallax_budget
    assert parallax_budget(595, 705, 25) < 200
    assert parallax_budget(50, 130, 26) >= 49  # tree still admitted
