"""The ground cue must recover a boundary it was not shown, and must lose to a
control that has no information.

A test that only checks "the error is small" proves nothing: a cue that answers
"everything low in frame is ground" also scores small on frames where the
boundary really is low. So the discriminating check is a CONTROL fitted on the
same images with the boundaries replaced by noise -- if that does not score
worse, the test is measuring the row prior rather than the boundary, and the
assertion below is the only reason to believe the number.
"""

import numpy as np

from steadycut.framing.ground_cue import GroundCue

H, W = 240, 320


def synth(boundary_at, seed=0):
    """A frame with a known ground/forest boundary: litter below, forest above."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    xs = np.arange(W)
    row = (boundary_at + 0.04 * np.sin(xs / 40.0)) * H       # slightly curved
    ground = np.arange(H)[:, None] > row[None, :]

    # forest: dark green with a few vertical trunks
    forest = np.zeros((H, W, 3), np.int16)
    forest[..., 0] = 38 + rng.integers(0, 8, (H, W))         # b
    forest[..., 1] = 58 + rng.integers(0, 8, (H, W))         # g
    forest[..., 2] = 42 + rng.integers(0, 8, (H, W))         # r
    for trunk_x in range(10, W, 57):
        forest[:, trunk_x:trunk_x + 4] = (70, 62, 58)

    # ground: bright, rough litter
    litter = np.zeros((H, W, 3), np.int16)
    litter[..., 0] = 96 + rng.integers(0, 40, (H, W))
    litter[..., 1] = 126 + rng.integers(0, 40, (H, W))
    litter[..., 2] = 152 + rng.integers(0, 40, (H, W))

    img = np.where(ground[..., None], litter, forest).astype(np.uint8)
    return img, row / H


def error(cue, img, truth):
    pred = cue.boundary(img)
    ok = np.isfinite(pred)
    return float(np.nanmean(np.abs(pred[ok] - truth[ok]))) if ok.any() else float("inf")


def test_recovers_a_held_out_boundary():
    samples = [synth(0.30 + 0.06 * i, seed=i) for i in range(3)]
    cue = GroundCue.fit(samples)
    img, truth = synth(0.58, seed=99)                        # not in the fit
    e = error(cue, img, truth)
    assert e < 0.06, f"held-out boundary error {e:.3f} of frame height"


def test_beats_a_mislabeled_control():
    samples = [synth(0.30 + 0.06 * i, seed=i) for i in range(3)]
    good = GroundCue.fit(samples)
    rng = np.random.default_rng(7)
    noisy = [(img, rng.uniform(0.25, 0.65, size=W)) for img, _ in samples]
    bad = GroundCue.fit(noisy)

    img, truth = synth(0.58, seed=99)
    e_good, e_bad = error(good, img, truth), error(bad, img, truth)
    assert e_bad > 2.0 * e_good, (
        f"control is not worse (good {e_good:.3f}, control {e_bad:.3f}): the "
        f"test would pass on a cue that only knows 'low in frame is ground'")


def test_survives_shade_where_colour_alone_does_not():
    """The measured failure, in a test: same colour above and below, differing
    only in texture and structure.

    On real footage colour dies in shade -- the forest floor becomes the same
    brown as the trail, measured at 0.66-0.82 coverage in EVERY row of frames
    where a human sees the boundary plainly. Synthetic frames with clean colours
    cannot express that, so the earlier tests pass a colour-only cue happily.
    Here the two regions share a colour and differ in roughness (litter) versus
    vertical structure (trunks), which is the pair the features exist to use.
    """
    rng = np.random.default_rng(3)
    xs = np.arange(W)
    row = (0.5 + 0.03 * np.sin(xs / 50.0)) * H
    ground = np.arange(H)[:, None] > row[None, :]

    # one shared brown, so colour carries no information at all
    base = np.stack([np.full((H, W), 110), np.full((H, W), 118),
                     np.full((H, W), 124)], axis=-1).astype(np.int16)
    rough = base + rng.integers(-45, 45, (H, W, 3))          # litter: rough
    smooth = base + rng.integers(-3, 3, (H, W, 3))           # floor: smooth
    img = np.where(ground[..., None], rough, smooth)
    for trunk_x in range(12, W, 61):                          # trunks: structure
        img[:, trunk_x:trunk_x + 3] -= 70
    img = np.clip(img, 0, 255).astype(np.uint8)
    truth = row / H

    from steadycut.framing import ground_cue as G
    samples = [synth(0.30 + 0.06 * i, seed=i) for i in range(3)]
    full = G.GroundCue.fit(samples + [(img, truth)])
    e_full = error(full, img, truth)

    orig = G.features
    G.features = lambda im: orig(im)[..., 0:3]                # colour only
    try:
        colour_only = G.GroundCue.fit(samples + [(img, truth)])
        e_colour = error(colour_only, img, truth)
    finally:
        G.features = orig

    assert e_full < 0.10, f"full cue failed in shade: {e_full:.3f}"
    assert e_colour > e_full, (
        f"colour-only is not worse here ({e_colour:.3f} vs {e_full:.3f}), so this "
        f"test is not exercising the shade failure it exists for")


def test_boundary_is_nan_where_there_is_no_boundary():
    """A column with no ground reports NaN rather than a row to interpolate."""
    samples = [synth(0.30 + 0.06 * i, seed=i) for i in range(3)]
    cue = GroundCue.fit(samples)
    forest_only = np.zeros((H, W, 3), np.uint8)
    forest_only[..., 0], forest_only[..., 1], forest_only[..., 2] = 38, 58, 42
    b = cue.boundary(forest_only)
    assert np.isnan(b).mean() > 0.5, (
        "a frame with no ground should be mostly NaN, not a confident row")
