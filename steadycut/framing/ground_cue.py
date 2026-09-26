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
