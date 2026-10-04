"""Assist-mode proposals: three-model agreement as a labelling triage.

Three estimators produce a per-column ground boundary each: SegFormer
(semantics), DINOv2+kmeans (self-supervised clusters), Depth Anything
(depth clusters). On forest MTB they never all agree (0/30 windows); on
clean open snow they coincide to 0.004. Frames where all three agree
within AGREE go to the fit unlabeled-by-human; everything else goes to
the rider's Tk queue with the SegFormer boundary drawn as the proposal.

All torch/transformers/onnxruntime/sklearn imports live inside functions:
base steadycut must import without the `ml` extra installed. The depth
ONNX takes 0-255 input WITHOUT the /255 normalization -- that looks like
a bug and is not one; it is the exact convention the Task-8 probes
validated, and "fixing" it silently changes every boundary.
"""
from __future__ import annotations

import numpy as np

AGREE = 0.03           # max median pairwise disagreement to trust a frame
SLOPE_MAX_PX = 1.73    # tan(60 deg): drop steeper pseudo-boundary segments
GROUND_CLASSES = frozenset({6, 9, 13, 29, 52, 91})
# road grass earth field path dirt-track (ADE20K ids). No snow class exists;
# clean groomed snow lands in earth/path and agrees anyway (ski60 d=0.004).


def topmost(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Topmost True pixel per column, NaN where the column has none."""
    out = np.full(width, np.nan)
    for c in range(width):
        rows = np.flatnonzero(mask[:, c])
        if len(rows):
            out[c] = rows[0] / height
    return out


def agreement(*bounds: np.ndarray) -> float:
    """Worst median pairwise disagreement over jointly finite columns.

    NaN when fewer than half the columns are jointly finite (no common
    cover is itself a refusal, e.g. cluttered resort frames).
    """
    joint = np.ones_like(bounds[0], dtype=bool)
    for b in bounds:
        joint &= np.isfinite(b)
    if joint.sum() < len(bounds[0]) // 2:
        return float("nan")
    worst = 0.0
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            worst = max(worst, float(np.median(
                np.abs(bounds[i][joint] - bounds[j][joint]))))
    return worst


def slope_keep(boundary: np.ndarray, height: int,
               max_px: float = SLOPE_MAX_PX) -> np.ndarray:
    """Columns whose segment is flatter than 60 degrees.

    Near-vertical runs are trunk dives and sky jumps, never the far
    boundary (measured: the dives that poisoned the raw fit). Comparison
    is in px per column -- the boundary is in frame fractions, so scale
    by height first; comparing fractions to 1.73 keeps everything and
    the filter silently does nothing.
    """
    keep = np.zeros(len(boundary), dtype=bool)
    idx = np.flatnonzero(np.isfinite(boundary))
    if len(idx) == 0:
        return keep
    keep[idx[0]] = True
    for k in range(1, len(idx)):
        if idx[k] == idx[k - 1] + 1:
            if abs(boundary[idx[k]] - boundary[idx[k - 1]]) * height <= max_px:
                keep[idx[k]] = True
    return keep


def median_boundary(*bounds: np.ndarray) -> np.ndarray:
    """Per-column median across estimators (NaN-aware)."""
    return np.nanmedian(np.stack(bounds), axis=0)


class AssistModels:
    """The three estimators, loaded once via the pinned manifest."""

    def __init__(self, models_dir=None):
        from steadycut import models as _models
        self._models = _models
        self._dir = models_dir
        self._seg = None
        self._dino = None
        self._depth = None

    def seg(self):
        if self._seg is None:
            import torch
            from transformers import SegformerForSemanticSegmentation
            d = self._models.ensure("segformer-b2-ade", self._dir)
            self._torch = torch
            self._seg = SegformerForSemanticSegmentation.from_pretrained(
                str(d)).eval()
        return self._seg

    def dino(self):
        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel
            d = self._models.ensure("dinov2-small", self._dir)
            self._dino_proc = AutoImageProcessor.from_pretrained(str(d))
            self._dino = AutoModel.from_pretrained(str(d)).eval()
        return self._dino

    def depth_session(self):
        if self._depth is None:
            import onnxruntime as ort
            d = self._models.ensure("depth-anything-v2-small-onnx",
                                    self._dir)
            files = list(d.glob("model_quantized.onnx"))
            if not files:
                from steadycut.models import ModelUnavailable
                raise ModelUnavailable(
                    "depth onnx missing model_quantized.onnx")
            self._depth = ort.InferenceSession(
                str(files[0]), providers=["CPUExecutionProvider"])
        return self._depth

    def boundaries(self, bgr: np.ndarray):
        """(seg, dino, depth) per-column boundaries for one BGR frame."""
        import cv2
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # segformer
        r = cv2.resize(rgb.astype(np.float32), (512, 512),
                       interpolation=cv2.INTER_CUBIC)
        torch = self._torch
        with torch.no_grad():
            xs = torch.from_numpy(
                ((r / 255.0 - mean) / std).transpose(2, 0, 1)[None])
            logits = self.seg()(pixel_values=xs).logits[0]
            sl = torch.nn.functional.interpolate(
                logits.unsqueeze(0), size=(h, w), mode="bilinear",
                align_corners=False).argmax(1)[0].numpy()
        b_seg = topmost(np.isin(sl, list(GROUND_CLASSES)), h, w)
        # dino + kmeans, widest bottom-edge cluster
        with torch.no_grad():
            xd = self._dino_proc(
                images=rgb, return_tensors="pt")["pixel_values"]
            feat = self.dino()(pixel_values=xd
                               ).last_hidden_state[0, 1:, :].numpy()
        from sklearn.cluster import KMeans
        gh = gw = int(np.sqrt(len(feat)))
        dl = KMeans(n_clusters=4, n_init=4,
                    random_state=0).fit_predict(feat).reshape(gh, gw)
        full = cv2.resize(dl.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(int)
        counts = np.bincount(full[-1], minlength=4)
        b_dino = topmost(full == int(counts.argmax()), h, w)
        # depth + kmeans on log depth, bottom-center cluster.
        # 0-255 WITHOUT /255: the validated convention (see module doc).
        r3 = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
        x3 = ((r3.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
        sess = self.depth_session()
        dd = np.asarray(sess.run(
            None, {sess.get_inputs()[0].name: x3})[0])[0]
        dh = cv2.resize(dd, (w, h), interpolation=cv2.INTER_CUBIC)
        z = np.log(np.maximum(dh, 1e-6)).reshape(-1, 1)
        lab = KMeans(n_clusters=3, n_init=4,
                     random_state=0).fit_predict(z).reshape(h, w)
        b_depth = topmost(lab == lab[int(h * 0.95), w // 2], h, w)
        return b_seg, b_dino, b_depth


def propose(frame: np.ndarray, models: AssistModels):
    """A labelling proposal: (boundary, trusted) for one BGR frame.

    Trusted frames (triple agreement) go to the fit as-is; untrusted ones
    go to the rider with the slope-filtered SegFormer boundary drawn. The
    SegFormer boundary is the proposal either way -- it was the closest
    single model (0.085 raw) -- but only agreement spends rider trust.
    """
    b_seg, b_dino, b_depth = models.boundaries(frame)
    h, w = frame.shape[:2]
    a = agreement(b_seg, b_dino, b_depth)
    trusted = bool(np.isfinite(a) and a <= AGREE)
    if trusted:
        return median_boundary(b_seg, b_dino, b_depth), True
    keep = slope_keep(b_seg, h)
    out = np.full(w, np.nan)
    out[keep] = b_seg[keep]
    return out, False
