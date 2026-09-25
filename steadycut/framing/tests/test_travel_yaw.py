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


def args_for(travel_yaw):
    return render("in.insv", "out.mp4",
                  [P.PathPoint(t=0.0, fov=100.0, pitch=-22.0)],
                  travel_yaw=travel_yaw, dry_run=True)


def filtergraph(argv):
    return argv[argv.index("-filter_complex") + 1]


def test_default_is_single_pass():
    """A normally mounted camera must not pay for a second reprojection."""
    graph = filtergraph(args_for(0.0))
    assert graph.count("v360=") == 1
    assert "input=dfisheye" in graph
    assert "input=equirect" not in graph


def test_reversed_mount_reorients_the_sphere():
    graph = filtergraph(args_for(180.0))
    assert graph.count("v360=") == 2
    # The first pass turns the fisheye into a sphere with the reorientation...
    first, second = graph.split(";")
    assert "input=dfisheye" in first and "output=equirect" in first
    assert "yaw=-180" in first
    # ...and the second projects from that sphere, not from the fisheye again.
    assert "input=equirect" in second
    # id_fov describes the fisheye input and must not be passed to a sphere.
    assert "id_fov=0" in second


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