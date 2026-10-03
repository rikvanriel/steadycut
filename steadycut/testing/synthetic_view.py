"""Predict the rendered view the way the render actually builds it.

A (yaw, pitch, roll) command acts on sphere content as
R_z(roll) R_x(pitch) R_y(-yaw) -- yaw negated, pitch and roll as given.
That is MEASURED on real renders (single markers per axis, then all sign and
order combinations on fresh angles: 0.8 px median against 25+ px for
everything else), not read off the filter source. A plain reading of
vf_v360.c `calculate_rotation` predicts R_y R_x R_z with no flips, and the
renders disagree with that reading; the discrepancy is documented in
`to_euler_zxy` and not resolved, so do not "simplify" this toward the source
without re-running the render test.

The matrices below are built explicitly: scipy's lowercase Euler sequences
compose in the REVERSE order of what the names suggest (lowercase 'yxz'
gives R_z R_x R_y), and that single confusion produced two wrong "measured"
conventions in a row.

For the static synthetic source the rendered view is exactly the path's own
angles through that composition. There is no camera attitude in the prediction:
the source contains no camera motion, so composing q_raw in -- as an earlier
version did -- scores ~150 px where this scores ~1.3.
"""
import numpy as np


def v360_matrix(yaw_deg, pitch_deg, roll_deg):
    """3x3 matrix of how a v360 command acts on sphere content: MEASURED.

    R_z(roll) R_x(pitch) R_y(-yaw): yaw negated, pitch and roll as given.
    Accepts scalars or arrays; arrays broadcast to (N, 3, 3).
    """
    y, p, r = np.radians(np.broadcast_arrays(-np.asarray(yaw_deg, float),
                                             np.asarray(pitch_deg, float),
                                             np.asarray(roll_deg, float)))
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    z = np.zeros_like(y)
    o = np.ones_like(y)
    ry = np.stack([np.stack([cy, z, sy], -1),
                   np.stack([z, o, z], -1),
                   np.stack([-sy, z, cy], -1)], -2)
    rx = np.stack([np.stack([o, z, z], -1),
                   np.stack([z, cp, -sp], -1),
                   np.stack([z, sp, cp], -1)], -2)
    rz = np.stack([np.stack([cr, -sr, z], -1),
                   np.stack([sr, cr, z], -1),
                   np.stack([z, z, o], -1)], -2)
    return rz @ rx @ ry


def predict_pixels(dirs_sphere, yaw, pitch, roll):
    """Marker directions -> camera-frame directions under the path's angles."""
    m = v360_matrix(np.asarray(yaw, float), np.asarray(pitch, float),
                    np.asarray(roll, float))
    return np.asarray(dirs_sphere, float) @ m[..., :].swapaxes(-1, -2)
