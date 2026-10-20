"""factory.py pins: layout decoded from Jan26 file's known values."""
from steadycut.ingest.cameras.factory import lens_circle, lens_quat

SRC = ('/backup/insta360/VID_20260126_144450_00_010-Original/'
       'VID_20260126_144450_00_010.insv')


def test_factory_layout_pins():
    from steadycut.ingest.cameras.factory import factory_geometry
    g = factory_geometry(SRC)
    assert g['serial'] == 'IBMYA2410M3EB5'
    assert abs(g['rolling_shutter_s'] - 0.008957) < 1e-6
    c0 = lens_circle(g['offset_v2'], 0)
    assert abs(c0['cx'] - 2899.71) < 0.01
    assert abs(c0['cy'] - 4011.72) < 0.01
    assert abs(c0['r'] - 2991.85) < 0.01
    assert abs(c0['half_fov_deg'] - 91.065) < 1e-3
    c1 = lens_circle(g['offset_v2'], 1)
    assert abs(c1['cx'] - 2904.47) < 0.01
    assert abs(c1['half_fov_deg'] - 90.3) < 1e-3
    q0 = lens_quat(g['offset_v2'], 0)
    assert abs(q0[0] - 1.0) < 1e-9
    assert abs(q0[2] - 0.35412037) < 1e-6
    assert g['pts_type'] == 2


def test_fit_lens_circle_on_synthetic():
    import numpy as np
    from steadycut.ingest.cameras.factory import fit_lens_circle
    img = np.zeros((600, 600), np.uint8)
    cv2 = __import__('cv2')
    cv2.circle(img, (295, 307), 280, 200, -1)
    c = fit_lens_circle(img)
    assert abs(c['cx'] - 295) < 3
    assert abs(c['cy'] - 307) < 3
    assert abs(c['r'] - 280) < 6  # fixed dark threshold biases r out
    # by ~2 blur-sigma; center (the load-bearing output) stays tight
    assert c['n'] > 1000


def test_angular_finger_mask_is_sector_not_rows():
    import numpy as np
    from steadycut.ingest.cameras.plumb import angular_finger_mask
    m = angular_finger_mask(200, 200, 100, 100, 90)
    assert m.shape == (200, 200)
    assert m[190, 100]  # bottom rim masked
    assert not m[100, 100]  # center kept
    assert not m[10, 100]  # top kept


def test_chain_edgels_synthetic():
    import numpy as np
    from steadycut.ingest.cameras.plumb import chain_edgels
    cv2 = __import__('cv2')
    img = np.zeros((300, 300), np.uint8)
    cv2.line(img, (20, 150), (280, 150), 255, 3)
    chains = chain_edgels(img, min_len=100)
    assert len(chains) >= 1
    assert max(len(c) for c in chains) > 200
