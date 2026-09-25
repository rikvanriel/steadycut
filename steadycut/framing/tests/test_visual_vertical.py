"""Known-answer test for recovering the world vertical from an image.

This is the version of the test that can actually fail, and that is the point. The
earlier world-pitch work shipped a test built from the SAME assumed conventions as the
code it checked, so it passed while the real render came out inverted. Here the camera
orientation is CHOSEN, the scene is projected from it, and the estimator must recover
what was chosen. If it cannot, the method does not work, whatever the real-data numbers
look like.

The scene: gravity-aligned vertical lines at known positions, which is what tree trunks
approximate. Projected through a known rotation - pitch and roll - with a known focal
length, which a rendered flat view at a stated field of view provides exactly.

The metric is the angle between the direction the estimator recovers and the direction
that was used to build the image. Sign conventions cannot hide in that: the direction
is compared against the truth that generated it, not against another assumption.
"""
import numpy as np

from steadycut.framing import visual_vertical


def world_up_in_camera(pitch_deg, roll_deg):
    """The world up direction expressed in the camera frame.

    The camera is pitched down by `pitch_deg` (so a positive value looks down) and then
    rolled by `roll_deg` about its own optical axis. Image y points DOWN, so the
    camera's y axis is the world's -y.
    """
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    # Camera-from-world: pitch down about world x, then roll about the optical axis.
    rp = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    rr = np.array([[np.cos(r), 0, -np.sin(r)],
                   [0, 1, 0],
                   [np.sin(r), 0, np.cos(r)]])
    cam_from_world = rr @ rp
    return cam_from_world @ np.array([0.0, 1.0, 0.0])


def render_scene(pitch_deg, roll_deg, w=1440, h=1080, h_fov=120.0,
                 n_lines=60, seed=1):
    """An image of gravity-aligned vertical lines at a known orientation."""
    import cv2
    f = (w / 2.0) / np.tan(np.radians(h_fov / 2.0))
    cx, cy = w / 2.0, h / 2.0
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.uint8)

    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    rp = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rr = np.array([[np.cos(r), 0, -np.sin(r)], [0, 1, 0], [np.sin(r), 0, np.cos(r)]])
    cam_from_world = rr @ rp

    def project(pt_world):
        v = cam_from_world @ pt_world
        if v[2] < 0.3:
            return None
        return (cx + f * v[0] / v[2], cy + f * v[1] / v[2])

    for _ in range(n_lines):
        x0 = rng.uniform(-6.0, 6.0)
        z0 = rng.uniform(2.0, 25.0)
        y0 = rng.uniform(-3.0, 3.0)
        top = project(np.array([x0, y0 + 8.0, z0]))
        bot = project(np.array([x0, y0 - 8.0, z0]))
        if top is None or bot is None:
            continue
        a = (int(np.clip(top[0], -5 * w, 5 * w)), int(np.clip(top[1], -5 * h, 5 * h)))
        b = (int(np.clip(bot[0], -5 * w, 5 * h)), int(np.clip(bot[1], -5 * h, 5 * h)))
        cv2.line(img, a, b, 255, 3)
    return img


def test_recovers_a_known_orientation():
    """The estimator must recover pitch and roll it was not told about."""
    for pitch in (-10.0, -20.0, -35.0):
        for roll in (0.0, 8.0, -12.0):
            img = render_scene(pitch, roll)
            vp, used, total = visual_vertical.vertical_vanishing_point(img)
            assert vp is not None, f"no vanishing point at pitch {pitch} roll {roll}"
            w, h = img.shape[1], img.shape[0]
            f = (w / 2.0) / np.tan(np.radians(120.0 / 2.0))
            d = np.array([(vp[0] - w / 2.0) / f, (vp[1] - h / 2.0) / f, 1.0])
            d /= np.linalg.norm(d)
            truth = world_up_in_camera(pitch, roll)
            truth = truth / np.linalg.norm(truth)
            err = np.degrees(np.arccos(np.clip(abs(float(np.dot(d, truth))), -1, 1)))
            assert err < 2.0, (
                f"pitch {pitch} roll {roll}: recovered direction is {err:.2f} deg off"
                f" ({used} lines of {total})")