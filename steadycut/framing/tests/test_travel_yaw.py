"""travel_yaw must rotate the sphere, not offset the view.

Why this matters: v360's pitch is degenerate at yaw 180. A camera mounted the
other way round has its direction of travel at yaw 180 in the sphere, so if the
parameter were implemented by offsetting the view yaw, the view would land at yaw
180 and pitch would stop working for the one direction the whole project is about.

Rotating the sphere instead keeps the framing yaw near 0, so tilt behaves the same
regardless of how the camera was mounted.

These checks read the generated ffmpeg arguments, so they are fast and do not
render. The behavioural half - that pitch still has an effect at travel_yaw=180 -
was verified against real footage: the image difference between pitch -22 and -40
is 39.5 of 255, against 0.25 for the degenerate case.
"""
import pytest

from steadycut.stabilization import path as P
from steadycut.render.reframe import render


def args_for(travel_yaw, method="blend"):
    from steadycut.render.stitch import StitchSpec
    spec = StitchSpec(method=method, gain=(1.0, 1.0, 1.0))
    return render("in.insv", "out.mp4",
                  [P.PathPoint(t=0.0, fov=100.0, pitch=-22.0)],
                  travel_yaw=travel_yaw, dry_run=True, stitch=spec)


def filtergraph(argv):
    return argv[argv.index("-filter_complex") + 1]


def test_default_is_blend_stitch_then_single_flat():
    """Default sphere assembly is the two-pass blend (one v360 per lens),
    then a single flat projection reading that sphere."""
    graph = filtergraph(args_for(0.0))
    assert graph.count("v360=") == 3
    assert "maskedmerge" in graph
    assert "input=dfisheye" not in graph
    assert graph.split(";")[-1].endswith("[v]")
    assert "input=equirect" in graph.split(";")[-1]


def test_reversed_mount_reorients_the_sphere():
    graph = filtergraph(args_for(180.0))
    # Blend stitch (2 lens passes) + sphere reorientation + flat projection.
    assert graph.count("v360=") == 4
    assert "maskedmerge" in graph
    assert "yaw=-180" in graph
    # ...and the final projection reads from a sphere, not the fisheye.
    assert "input=equirect" in graph.split(";")[-1]
    # id_fov describes the fisheye input and must not be passed to a sphere.
    assert "id_fov=0" in graph.split(";")[-1]


def test_framing_yaw_is_unchanged_by_reorientation():
    """The reason for rotating the sphere: the framing stays near yaw 0."""
    for travel_yaw in (0.0, 180.0):
        graph = filtergraph(args_for(travel_yaw))
        assert "yaw=0.0:pitch=-22.0" in graph.split(";")[-1]


def test_sphere_size_is_configurable():
    argv = render("in.insv", "out.mp4",
                  [P.PathPoint(t=0.0, fov=100.0)], travel_yaw=90.0,
                  sphere_size=(1024, 512), dry_run=True)
    assert "w=1024:h=512" in filtergraph(argv)