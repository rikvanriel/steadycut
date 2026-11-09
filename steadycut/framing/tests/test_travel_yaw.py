"""travel_yaw offsets the view yaw; there is no sphere-rotation pass.

A camera mounted the other way round has its direction of travel at yaw 180
in the sphere. The old design rotated the sphere first (a full extra equirect
render) on the theory that v360's pitch is degenerate at yaw 180. Re-measured
on ffmpeg 8.1.3, pitch works fine at yaw 180 (71.3 vs 45.9 mean pixel diff at
yaw 180 vs 0, static==sendcmd to 0.00), so the framing is a plain view-yaw
offset of -travel_yaw and the second pass is gone.

The behavioural half - that pitch still has an effect at travel_yaw=180 - is
covered against real footage by test_reframe_traps.py::test_pitch_is_honoured_at_yaw_180.
These checks read the generated ffmpeg arguments, so they are fast and do not
render.
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
    assert graph.count("v360=") + graph.count("v360@view=") == 3
    assert "maskedmerge" in graph
    assert "input=dfisheye" not in graph
    assert graph.split(";")[-1].endswith("[v]")
    assert "input=equirect" in graph.split(";")[-1]


def test_reversed_mount_offsets_the_view_yaw():
    graph = filtergraph(args_for(180.0))
    # Blend stitch (2 lens passes) + one flat projection, no sphere
    # reorientation stage in between.
    assert graph.count("v360=") + graph.count("v360@view=") == 3
    assert "maskedmerge" in graph
    assert "output=equirect:id_fov=" not in graph
    # ...and the final projection looks along the mount-corrected yaw,
    # with the pitch channel intact.
    assert "input=equirect" in graph.split(";")[-1]
    assert "yaw=-180.0:pitch=-22.0" in graph.split(";")[-1]
    # id_fov describes the fisheye input and must not be passed to a sphere.
    assert "id_fov=0" in graph.split(";")[-1]


def test_folded_view_yaw_follows_the_mount():
    """The framing yaw is the path yaw minus travel_yaw, wrapped to range."""
    for travel_yaw, want in ((0.0, "yaw=0.0:pitch=-22.0"),
                             (180.0, "yaw=-180.0:pitch=-22.0"),
                             (90.0, "yaw=-90.0:pitch=-22.0")):
        graph = filtergraph(args_for(travel_yaw))
        assert want in graph.split(";")[-1], (travel_yaw, graph)


def test_sphere_size_is_configurable():
    from steadycut.render.stitch import StitchSpec
    argv = render("in.insv", "out.mp4",
                  [P.PathPoint(t=0.0, fov=100.0)], travel_yaw=90.0,
                  sphere_size=(1024, 512), dry_run=True,
                  stitch=StitchSpec(method="blend", gain=(1.0, 1.0, 1.0)))
    assert "w=1024:h=512" in filtergraph(argv)