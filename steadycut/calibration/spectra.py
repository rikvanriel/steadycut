"""Where does the head-motion energy live?

The cutoff in path.py cancels everything faster than 1/cutoff_s, so the useful
question is which frequencies actually carry the angular motion.  Integrating
the rate first gives the angle, whose spectrum is what the correction acts on:
a given rate amplitude produces angle amplitude falling as 1/f.
"""
import sys

import numpy as np

from steadycut.ingest.telemetry import read_telemetry


def welch(x, rate, seg=4.0, overlap=0.5):
    """Power spectrum, numpy only.  Returns freq, psd."""
    n = int(seg * rate)
    step = int(n * (1 - overlap))
    if x.size < n:
        n = x.size
        step = n
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    count = 0
    for start in range(0, x.size - n + 1, step):
        chunk = (x[start:start + n] - x[start:start + n].mean()) * win
        acc += np.abs(np.fft.rfft(chunk)) ** 2
        count += 1
    if not count:
        return np.zeros(1), np.zeros(1)
    psd = acc / count / (rate * np.sum(win ** 2))
    freq = np.fft.rfftfreq(n, 1.0 / rate)
    return freq, psd


def band(t, gyro):
    """Geometric mean of the 10-90% energy band of the rotation rate.

    Power is summed per axis rather than over the magnitude of the rate vector.
    Taking the magnitude rectifies each axis, and the absolute value of a
    sinusoid oscillates at twice its frequency: a pure 5 Hz tone measured this
    way reports as 10 Hz, which biases the band upward by up to a factor of two.
    """
    fs = 1.0 / np.median(np.diff(t))
    n = gyro.shape[0]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    centred = gyro - gyro.mean(axis=0)
    win = np.hanning(n)[:, None]
    p = (np.abs(np.fft.rfft(centred * win, axis=0)) ** 2).sum(axis=1)
    p[f < 0.3] = 0.0
    c = np.cumsum(p) / p.sum()
    lo = float(f[np.searchsorted(c, 0.10)])
    hi = float(f[np.searchsorted(c, 0.90)])
    return float(np.sqrt(lo * hi)), lo, hi


def median_freq(t, gyro):
    """Frequency that splits the rotation-rate energy in half.

    This is the predictor for the optimal cutoff. The geometric mean of the
    10-90 percent band was tried first (as `band`) and is not the predictor.
    """
    fs = 1.0 / np.median(np.diff(t))
    n = gyro.shape[0]
    f = np.fft.rfftfreq(n, 1.0 / fs)
    centred = gyro - gyro.mean(axis=0)
    p = (np.abs(np.fft.rfft(centred * np.hanning(n)[:, None], axis=0)) ** 2).sum(axis=1)
    p[f < 0.3] = 0.0
    c = np.cumsum(p) / p.sum()
    return float(f[np.searchsorted(c, 0.5)])


def analyse(path, label, start=400.0, dur=30.0, axes=("yaw", "pitch", "roll")):
    imu = read_telemetry(path)
    w = (imu.time_s >= start) & (imu.time_s <= start + dur)
    t = imu.time_s[w]
    rate = 1.0 / np.median(np.diff(t))
    order = {1: "yaw", 2: "pitch", 0: "roll"}
    print(f"\n{label}  ({rate:.0f} Hz, {t.size} samples, {dur:.0f}s)")
    for axis, name in order.items():
        # integrate rate -> angle so the spectrum is in degrees, not deg/s
        dt = np.gradient(t)
        ang = np.cumsum(imu.gyro[w][:, axis] * dt)
        ang = ang - ang.mean()
        f, p = welch(ang, rate)
        band = (f > 0.1)
        total = np.trapezoid(p[band], f[band])
        # Angle variance is dominated by the lowest frequencies simply
        # because integrating a rate noise floor multiplies the spectrum by
        # 1/f^2.  So also report the RATE spectrum, which answers the actual
        # question: is there vibration energy at high frequency?
        if total <= 0:
            continue
        for cut in (2.0, 5.0, 10.0):
            hi = (f > cut)
            frac = np.trapezoid(p[hi], f[hi]) / total
            print(f"    {name:5s} above {cut:4.1f} Hz: {100*frac:5.1f}% of angle variance")
        # dominant frequency
        peak = f[band][np.argmax(p[band])]
        print(f"    {name:5s} peak at {peak:.2f} Hz, angle std {np.std(ang):.3f} deg")
        fr, pr = welch(imu.gyro[w][:, axis], rate)
        rb = fr > 0.1
        rt = np.trapezoid(pr[rb], fr[rb])
        parts = []
        for cut in (2.0, 10.0, 30.0):
            hi = fr > cut
            parts.append(f">{cut:.0f}Hz {100*np.trapezoid(pr[hi], fr[hi])/rt:4.1f}%")
        print(f"    {name:5s} RATE spectrum: " + "  ".join(parts)
              + f"   rate std {np.std(imu.gyro[w][:, axis]):5.1f} deg/s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m steadycut.calibration.spectra <file.insv> "
              "[more ...]", file=sys.stderr)
        raise SystemExit(2)
    for path in sys.argv[1:]:
        analyse(path, path.rsplit("/", 1)[-1])
