"""Pins for the blend stitch: cached artifacts, sane gains, wired graph."""
import numpy as np

from steadycut.render.stitch import (GAIN_DEFAULT, StitchSpec, cache_dir,
                                     deflare_frame, mask_path, track_gain_filter)


def test_mask_is_cached_and_binary_sized():
    import cv2
    m1 = mask_path(182.2, 4.0)
    m2 = mask_path(182.2, 4.0)
    assert m1 == m2 and m1.exists()
    img = cv2.imread(str(m1), 0)
    assert img.shape == (1024, 2048)
    # ramp, not a step: mid values exist between the plateaus
    assert 0 < int((img == 128).sum()) + int(((img > 10) & (img < 245)).sum())


def test_deflare_field_is_zero_at_center_and_grows_to_rim():
    import cv2
    p = deflare_frame(182.2, (5.39, -7.2, 9.36, -0.85))
    img = cv2.imread(str(p)).astype(np.float32)
    assert img[1440, 1440, 0] == 0.0
    rim = img[1440, 100, 0]
    assert 4.0 < rim < 12.0
    assert (img[:, :, 1:] == 0).all()


def test_gain_default_is_near_unity_and_bounded():
    assert all(0.9 < g < 1.1 for g in GAIN_DEFAULT)


def test_gain_filter_names_all_three_channels():
    f = track_gain_filter((1.016, 1.029, 1.043))
    assert "lutrgb=" in f and all(f"c='{c}" in f or f"{c}='" in f
                                  for c in "rgb")


def test_stitch_spec_defaults_are_bench_geometry():
    """Geometry defaults are the bench winner; the METHOD default is legacy
    for cost (the blend bake is minutes per window), so blend is opt-in."""
    s = StitchSpec()
    assert (s.lens_fov_a, s.lens_fov_b, s.ramp_deg) == (182.2, 180.6, 4.0)
    assert s.method == "legacy"
    assert StitchSpec(method="blend").method == "blend"
    assert s.gain is None  # measured per clip, never carried
