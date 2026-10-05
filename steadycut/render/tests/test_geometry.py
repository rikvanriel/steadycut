"""LensGeometry drives the filtergraph; explicit kwargs still win.

Every geometry field must END UP IN the filtergraph -- a parameter that
appears only in a signature does nothing (the input_fov lesson). These tests
grep the dry-run ffmpeg argument list for each field, so a field that stops
being wired fails here instead of silently rendering defaults.
"""
import math

from steadycut.ingest.cameras.insta360_x4 import PROFILE as X4
from steadycut.ingest.cameras.profiles import LensGeometry
from steadycut.render.reframe import ControlPoint, render, resolve_input


def _filtergraph(geometry=None, **kwargs) -> str:
    args = render("dummy.insv", "out.mp4", [ControlPoint(t=0.0, fov=100.0)],
                  geometry=geometry, dry_run=True, **kwargs)
    return next(a for a in args if "v360" in a)


def test_blend_stitch_graph_is_wired() -> None:
    """The blend path must actually reach the graph when asked for."""
    from steadycut.render.stitch import StitchSpec
    fg = _filtergraph(geometry=X4.lens,
                      stitch=StitchSpec(method="blend", gain=(1.0, 1.0, 1.0)))
    # Two-pass blend: per-lens equirect with calibrated FOVs, maskedmerge.
    assert "v360=input=fisheye:output=e:ih_fov=182.2" in fg
    assert "v360=input=fisheye:output=e:ih_fov=180.6" in fg
    assert "maskedmerge" in fg
    assert "reset_rot=1" in fg


def test_legacy_is_the_default_and_survives() -> None:
    from steadycut.render.stitch import StitchSpec
    fg = _filtergraph(geometry=X4.lens, stitch=StitchSpec(method="legacy"))
    assert fg.startswith("[0:v:1][0:v:0]hstack=inputs=2,")
    # sqrt(2) * 181.37, v360's diagonal convention, computed in resolve_input.
    assert "v360=input=dfisheye:output=flat:id_fov=256.49591380760825:" in fg
    assert "reset_rot=1" in fg


def test_front_first_flips_the_stream_mapping() -> None:
    from steadycut.render.stitch import StitchSpec
    g = LensGeometry(projection="dfisheye", lenses=2, lens_fov_deg=181.37,
                     stack_order="front_first")
    # Stream mapping is a legacy-graph property; pin it on the legacy path.
    assert _filtergraph(geometry=g,
                        stitch=StitchSpec(method="legacy")).startswith(
                            "[0:v:0][0:v:1]hstack=inputs=2,")


def test_single_lens_has_no_stack_stage() -> None:
    g = LensGeometry(projection="fisheye", lenses=1, lens_fov_deg=150.0,
                     stack_order="single")
    fg = _filtergraph(geometry=g)
    assert "hstack" not in fg
    assert "v360=input=fisheye" in fg
    assert f"id_fov={math.sqrt(2) * 150.0}" in fg


def test_explicit_kwargs_override_the_geometry() -> None:
    fg = _filtergraph(geometry=X4.lens, input_projection="fisheye",
                      input_fov=200.0, travel_yaw=180.0)
    assert "input=fisheye" in fg and "id_fov=200.0" in fg
    # travel_yaw forces the two-stage sphere reorientation.
    args = render("dummy.insv", "out.mp4", [ControlPoint(t=0.0, fov=100.0)],
                  geometry=X4.lens, travel_yaw=180.0, dry_run=True)
    fg2 = next(a for a in args if "equirect" in a and "v360" in a)
    assert "yaw=-180.0" in fg2


def test_travel_yaw_field_is_wired_from_the_geometry() -> None:
    g = LensGeometry(projection="dfisheye", lenses=2, lens_fov_deg=181.37,
                     stack_order="rear_first", travel_yaw=180.0)
    args = render("dummy.insv", "out.mp4", [ControlPoint(t=0.0, fov=100.0)],
                  geometry=g, dry_run=True)
    stage_one = next(a for a in args if "equirect" in a and "v360" in a)
    assert "yaw=-180.0" in stage_one


def test_resolve_input_priority() -> None:
    # explicit kwarg > geometry > X4-era default
    assert resolve_input(X4.lens)[1] == math.sqrt(2) * 181.37
    assert resolve_input(X4.lens, input_fov=200.0)[1] == 200.0
    assert resolve_input()[1] == math.sqrt(2) * 181.37
    assert resolve_input()[0] == "dfisheye"
    assert resolve_input(X4.lens, dual_stream=False)[2] is False
