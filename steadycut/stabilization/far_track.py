"""Step 1: far-field-only trajectory tracker.

Reuses rotate.rotation_between's ORB+RANSAC core but restricted to the FAR box
(parallax.FAR), so the bike/helmet can't pollute the estimate. Outputs per-frame
partial-affine (dx, dy, dtheta) + inlier count, and the accumulated trajectory.
No smoothing yet — that is step 2, judged against this raw trajectory.
"""
import subprocess
import numpy as np
import cv2
from steadycut.metrics import parallax

W, H = 1280, 960

def load(video):
    return parallax.frames(str(video), W, H)

def far_mask(h, w):
    y0, y1, x0, x1 = parallax.FAR
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 255
    return m

def track(video):
    # Streams frame pairs through the tracker: two frames alive at a time.
    # An earlier version decoded the whole clip to float32 first (150 frames
    # of 1280x960 = ~700MB plus ORB copies), which got the process OOM-killed
    # on a loaded box. Same estimates, ~50MB peak.
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    m = far_mask(H, W)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_ds = None, None
    traj = []  # (dx, dy, dtheta_deg, inliers)
    seen_first = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, ds = orb.detectAndCompute(img, m)
        if not seen_first:
            # The first frame has no preceding interval, so there is nothing to
            # measure yet -- and no row to emit. This must be an explicit flag:
            # a featureless frame yields ds=None, so testing prev_ds for None
            # would treat the frame AFTER it as a new first frame and drop a
            # second row.
            seen_first = True
            prev_kp, prev_ds = kp, ds
            continue
        if ds is None or len(kp) < 12 or len(prev_kp) < 12:
            # One row per INTERVAL, always. A frame with too few features to
            # match still gets a row (zero motion), because dropping it would
            # shift every later correction one frame against its frame -- and a
            # correction one frame late injects motion instead of removing it.
            traj.append((0.0, 0.0, 0.0, 0))
            prev_kp, prev_ds = kp, ds
            continue
        matches = matcher.match(prev_ds, ds)
        if len(matches) < 12:
            traj.append((0.0, 0.0, 0.0, 0))
            prev_kp, prev_ds = kp, ds
            continue
        pa = np.float32([prev_kp[mm.queryIdx].pt for mm in matches]).reshape(-1, 1, 2)
        pb = np.float32([kp[mm.trainIdx].pt for mm in matches]).reshape(-1, 1, 2)
        model, inliers = cv2.estimateAffinePartial2D(
            pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if model is None:
            traj.append((0.0, 0.0, 0.0, 0))
        else:
            dx, dy = float(model[0, 2]), float(model[1, 2])
            th = float(np.degrees(np.arctan2(model[1, 0], model[0, 0])))
            n_in = int(inliers.sum()) if inliers is not None else 0
            traj.append((dx, dy, th, n_in))
        prev_kp, prev_ds = kp, ds
        del frame, img
    cap.release()
    return np.array([(d[0], d[1], d[2]) for d in traj]), np.array([d[3] for d in traj])

if __name__ == "__main__":
    import sys
    video = sys.argv[1]
    deltas, inl = track(video)
    acc = np.cumsum(deltas, axis=0)
    print(f"{video}: {len(deltas)} steps, inliers median {np.median(inl):.0f} min {inl.min()}")
    print(f"  per-frame dx meanabs {np.abs(deltas[:,0]).mean():.2f}px bounce {np.diff(deltas[:,0]).std():.2f}px")
    print(f"  per-frame dy meanabs {np.abs(deltas[:,1]).mean():.2f}px bounce {np.diff(deltas[:,1]).std():.2f}px")
    print(f"  accumulated travel: dx {acc[-1,0]:+.1f}px dy {acc[-1,1]:+.1f}px dtheta {acc[-1,2]:+.2f}deg")
    np.save("/tmp/far_traj.npy", np.column_stack([acc, inl[:len(acc)]]))
    print("saved /tmp/far_traj.npy (acc_dx, acc_dy, acc_th, inliers)")
