"""Where the ground ends and the forest above it begins, for one recording.

A boundary between trail surface and the forest standing on it. Three cheaper
cues were measured dead on this footage before this one worked, and each died of
something specific, so the features here are chosen against those failures:

  * COLOUR alone cannot do it: in shade the forest floor is the same brown as the
    trail. Measured coverage was 0.66-0.82 in EVERY row of frames where the
    boundary is plainly visible to a human.
  * EDGE ORIENTATION alone cannot do it: the fisheye stretches the near field
    radially, which manufactures vertical structure at the bottom of frame,
    exactly where the trail is. The profile came out U-shaped and meaningless.
  * FLOW (inverse depth) cannot do it either, in three formulations (argmin on an
    equirect, argmin on a flat view, and a DP shortest path): the boundary is a
    SURFACE-ORIENTATION change, not a depth change, because litter continues into
    the distance between the trunks. All three landed mid-foliage or inside the
    ground.

What works is the same evidence, used JOINTLY and fitted per recording:

  r, g, b    the material
  texture    local gradient energy: litter is rough and isotropic, trunks are
             oriented, sky is flat
  row        the camera-frame pitch: the geometry prior, and the strongest single
             one available

That is five numbers per pixel and six parameters for the model, which is only
enough because nothing is asked to generalise across riders or lighting: the fit
is per recording, so the model learns thresholds, not geometry. Measured on
held-out frames of the recording it was fitted on, the boundary came back within
0.082 of frame height, against a spread of 0.3-0.6 between framings tonight.

The label set is the caller's business (see the plan's labelling pass): this
module takes rendered frames and a boundary per column, fits, and predicts. It
does not know where either came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARGIN = 0.05          # fraction of height left unlabelled around the boundary
BIKE_BAND = 0.08       # bottom of frame: the rider's own machine, not ground
SUB = 2                # pixel subsampling when fitting
ITERS = 400
LR = 0.5

# How far the cue may vary ACROSS COLUMNS before the frame is called
# structureless. The probability is compared column-to-column at a fixed row, so
# a frame whose ground/not-ground split comes only from the `row` prior scores
# zero here whatever its answer looks like. Calibrated so a real boundary on
# riding footage clears it with room and a flat frame fails it outright; it is
# a floor for "is the image doing any work", not a claim about any one frame.
X_SPREAD = 0.02


def features(img: np.ndarray) -> np.ndarray:
    """(h, w, 5) features from a BGR frame: r, g, b, texture, row."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.boxFilter(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), -1, (5, 5))
    h, w = gray.shape
    return np.stack([
        img[..., 2] / 255.0,                       # r
        img[..., 1] / 255.0,                       # g
        img[..., 0] / 255.0,                       # b
        tex / max(1e-6, float(np.percentile(tex, 95))),
        np.repeat((np.arange(h) / h)[:, None], w, axis=1),
    ], axis=-1)


@dataclass
class GroundCue:
    """A fitted boundary cue: weights plus the feature normalisation."""

    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def probability(self, img: np.ndarray) -> np.ndarray:
        """Per-pixel probability that the pixel is ground."""
        f = (features(img) - self.mean) / self.scale
        f = np.concatenate([f, np.ones(f.shape[:2] + (1,))], axis=-1)
        return 1.0 / (1.0 + np.exp(-f @ self.weights))

    def is_degenerate(self, img: np.ndarray, x_spread: float = X_SPREAD,
                      smooth: int = 31) -> str | None:
        """Why this frame has no boundary in it, or None if it has one.

        The cue's features include `row`, a pure vertical ramp that is identical
        in every frame -- a GEOMETRY PRIOR, not evidence. If the only thing
        separating ground from not-ground is that ramp, the boundary returned is
        the prior's, not the picture's. So the discriminator is built by asking
        the cue the same question TWICE: once with the image, and once with the
        image-derived features zeroed so only the prior and the bias remain. If
        the two answers agree, the image contributed nothing and the boundary
        came from geometry.

        The obvious alternative -- how much the probability varies ACROSS
        COLUMNS -- does not work, and the test that pinned this found it: a frame
        with a real boundary and a perfectly smooth sky above it is uniform
        across columns, so a column-variance test calls it structureless and
        refuses an honest frame. Comparing against the prior has no such blind
        spot, because it asks what the image added rather than how lumpy the
        image is.

        Measured on real footage: 0519a moves 0.286-0.348 across a 21 degree
        ladder and passes. 0519b returns 0.279 at every rung -- the same number
        to three decimals -- because the camera is pointed at the rider's own
        body rather than the trail, so no ground boundary is in view at all.

        The reason names the trail and the pointing deliberately: the rider has
        to be able to act on it, and "no boundary" alone does not say whether to
        re-frame, wait for better light, or suspect the tool.
        """
        f = (features(img) - self.mean) / self.scale
        f = np.concatenate([f, np.ones(f.shape[:2] + (1,))], axis=-1)
        if f.shape[0] < 2:
            return "the frame is a single row, so it cannot hold a boundary"
        # What the IMAGE contributes to the decision, with the row prior and the
        # bias taken out. If that contribution is the same at every pixel, the
        # image is adding a constant offset and no boundary is being found --
        # whatever the answer looks like. A flat frame of one colour does
        # exactly this, and so does a body filling the view.
        img_part = f[..., :4] @ self.weights[:4]
        if float(img_part.std()) < x_spread:
            return ("no ground boundary is visible in this frame: the image "
                    "adds nothing to the geometry prior, so there is no "
                    "boundary here to find -- the camera looks to be pointed at "
                    "something other than the trail (or pitched too far down at "
                    "the rider's own body). Re-frame, or set the pitch by eye.")
        return None

    @staticmethod
    def _boundary_from(p: np.ndarray, smooth: int = 31) -> np.ndarray:
        """The per-column topmost-ground row from a probability map."""
        pb = cv2.blur(p.astype(np.float32), (smooth, 1))
        h, w = pb.shape
        out = np.full(w, np.nan)
        for x in range(w):
            idx = np.nonzero(pb[:, x] > 0.5)[0]
            if idx.size:
                out[x] = idx[0] / h
        return out

    def boundary_or_none(self, img: np.ndarray, smooth: int = 31):
        """The boundary, or None when the frame has no boundary to return.

        `boundary` stays as it is because the fit needs a number even for a poor
        frame. This is the entry point a CRITERION wants, where a confident
        answer to an unanswerable frame is worse than no answer: the sweep reads
        NaN as unmeasurable and refuses, which is the honest outcome.
        """
        if self.is_degenerate(img, smooth=smooth) is not None:
            return None
        return self.boundary(img, smooth=smooth)

    def boundary(self, img: np.ndarray, smooth: int = 31) -> np.ndarray:
        """Per-column fraction of frame height where ground starts.

        NaN where the column holds no ground at all, which is the honest answer
        for a canopy frame and the input a caller's tilt rule needs: the horizon
        is not in this frame, do not guess a row for it.
        """
        p = cv2.blur(self.probability(img).astype(np.float32), (smooth, 1))
        ground = p > 0.5
        h, w = p.shape
        out = np.full(w, np.nan)
        for x in range(w):
            idx = np.nonzero(ground[:, x])[0]
            if idx.size:
                out[x] = idx[0] / h
        return out

    @classmethod
    def fit(cls, samples, margin: float = MARGIN,
            bike_band: float = BIKE_BAND) -> "GroundCue":
        """Fit from (image, boundary per column) pairs.

        `boundary` is a per-column fraction of height where ground starts, NaN
        where unknown. Pixels within `margin` of it are left unlabelled, and the
        bottom `bike_band` is never labelled ground: the rider's own machine is
        not ground, and calling it ground is how a cue learns the dark-mass
        confusion this project has already been bitten by twice.
        """
        X, y = [], []
        for img, boundary in samples:
            h, w = img.shape[:2]
            b = np.asarray(boundary, dtype=float)
            if b.shape != (w,):
                raise ValueError(f"boundary must have one value per column ({w})")
            known = np.isfinite(b)
            rows = np.repeat(np.arange(h)[:, None], w, axis=1)
            hi = np.where(known, b * h, np.nan)[None, :]
            above = known[None, :] & (rows < hi - margin * h)
            below = known[None, :] & (rows > hi + margin * h)
            below &= rows < (h * (1 - bike_band))
            f = features(img)
            for mask, val in ((above, 0.0), (below, 1.0)):
                sel = mask[::SUB, ::SUB]
                if not sel.any():
                    continue
                X.append(f[::SUB, ::SUB][sel])
                y.append(np.full(int(sel.sum()), val))
        if not X:
            raise ValueError("no labelled pixels: check the boundary arrays")
        X, y = np.concatenate(X), np.concatenate(y)

        mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xn = (X - mean) / scale
        Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
        weights = np.zeros(Xn.shape[1])
        for _ in range(ITERS):
            p = 1.0 / (1.0 + np.exp(-Xn @ weights))
            weights -= LR * (Xn.T @ (p - y)) / len(y)
        return cls(weights=weights, mean=mean, scale=scale)
