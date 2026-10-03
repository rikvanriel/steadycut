"""Predict the rendered view the way the render actually builds it.

Measured, not assumed: with one marker on the sphere and a static rotation,
v360's convention scores 0.4 px median for `zyx` Euler with the YAW negated,
against 101-314 px for every other candidate. v360's yaw is positive-clockwise
while its pitch and roll match the maths convention.

So the chain is: take the real path the pipeline built, convert each control
point's (yaw, pitch, roll) to a rotation under THAT convention, compose it
inside the sphere, and rotate the sphere with the raw camera attitude:

    view = q_raw * correction

which is not the correction, and not q_smooth either once the smoothing is
folded in -- it is the composition, and getting that wrong makes an exact
implementation look worse than a broken one.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def v360_rotation(yaw_deg, pitch_deg, roll_deg):
    """The rotation v360 builds from these three numbers.

    Measured convention: scipy 'zyx' with yaw negated. v360 composes
    R_yaw * R_pitch * R_roll about Y, X, Z, and its yaw runs clockwise.
    """
    return Rot.from_euler("zyx", [roll_deg, -yaw_deg, pitch_deg], degrees=True)


def view_at(q_raw_wxyz, yaw, pitch, roll):
    """World orientation of the view: q_raw composed with v360's correction."""
    q = np.asarray(q_raw_wxyz, float)
    w, x, y, z = q
    m = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return m @ v360_rotation(yaw, pitch, roll).as_matrix()


def camera_video(scene_path, out_path, q_raw, t, duration, fps, size,
                 h_fov, geometry=None, preset="ultrafast"):
    """Render the sphere as SEEN BY THE CAMERA, i.e. rotated by q_raw.

    This is the missing half of the substitution. A static equirect contains no
    camera rotation, so a path computed as `relative(q_raw, q_smooth)` has
    nothing to correct: the pipeline divides by a motion the video does not
    contain, and the expected view cannot be reconciled with the render at any
    time offset. Measured: ~165 px residual, flat across every frame-time
    convention from -40 ms to +40 ms, which is what a structural mismatch looks
    like rather than a timing one.

    Baking q_raw into the source makes the two cancel the way they do on real
    footage: view = q_raw * correction = q_smooth.
    """
    from steadycut.render.reframe import ControlPoint, render as _render

    e = Rot.from_quat(np.asarray(q_raw)[:, [1, 2, 3, 0]]).as_euler(
        "zyx", degrees=True)
    n = int(duration * fps)
    grid = np.arange(n) / fps
    pts = []
    for k in range(n):
        tc = (k + 0.5) / fps
        roll, neg_yaw, pitch = np.interp(
            tc, t, e[:, 0]), np.interp(tc, t, e[:, 1]), np.interp(tc, t, e[:, 2])
        pts.append(ControlPoint(t=k / fps, pitch=float(pitch),
                                yaw=float(-neg_yaw), roll=float(roll),
                                fov=h_fov))
    _render(scene_path, out_path, pts, size=size, input_projection="equirect",
            input_fov=360.0, dual_stream=False, start=0.0, duration=duration,
            preset=preset, geometry=geometry)


def predict_pixels(dirs_sphere, q_raw_wxyz, yaw, pitch, roll, h_fov, v_fov,
                   width, height):
    """Marker directions -> pixels under the measured convention."""
    m = view_at(q_raw_wxyz, yaw, pitch, roll)
    return dirs_sphere @ m.T