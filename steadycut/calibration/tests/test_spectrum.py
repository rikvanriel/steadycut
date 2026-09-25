"""Known-answer tests for the spectral band used to derive the cutoff."""
import numpy as np

from steadycut.calibration.spectra import band


def test_band_recovers_a_known_band() -> None:
    """Energy at 2 and 8 Hz must give a geometric mean of sqrt(2*8) = 4 Hz.

    The 10-90 percent band of two equal tones sits at the tones themselves, so
    the geometric mean of the band is the geometric mean of the tones. This is
    the check that the function measures frequency content rather than something
    correlated with it.
    """
    fs = 1000.0
    t = np.arange(0, 4.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 2.0 * t) + np.sin(2 * np.pi * 8.0 * t)
    gyro = np.stack([sig, np.zeros_like(sig), np.zeros_like(sig)], axis=1)
    gm, lo, hi = band(t, gyro)
    assert 3.4 < gm < 4.6, f"geometric mean {gm:.2f} Hz, expected about 4"
    assert 1.5 < lo < 2.5, f"lower edge {lo:.2f} Hz, expected about 2"
    assert 7.5 < hi < 8.5, f"upper edge {hi:.2f} Hz, expected about 8"


def test_band_ignores_drift() -> None:
    """A constant offset is not oscillation and must not enter the band."""
    fs = 1000.0
    t = np.arange(0, 4.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 5.0 * t) + 100.0
    gyro = np.stack([sig, np.zeros_like(sig), np.zeros_like(sig)], axis=1)
    gm, _, _ = band(t, gyro)
    assert 4.0 < gm < 6.5, f"geometric mean {gm:.2f} Hz, expected about 5"


def test_median_freq_recovers_a_single_tone() -> None:
    """The energy-halving frequency of one tone is that tone."""
    from steadycut.calibration.spectra import median_freq

    fs = 1000.0
    t = np.arange(0, 4.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 4.0 * t)
    gyro = np.stack([sig, np.zeros_like(sig), np.zeros_like(sig)], axis=1)
    med = median_freq(t, gyro)
    assert 3.4 < med < 4.6, f"median {med:.2f} Hz, expected about 4"


def test_median_freq_lands_between_two_equal_tones() -> None:
    """With equal energy at 2 and 8 Hz the halving point sits between them."""
    from steadycut.calibration.spectra import median_freq

    fs = 1000.0
    t = np.arange(0, 4.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 2.0 * t) + np.sin(2 * np.pi * 8.0 * t)
    gyro = np.stack([sig, np.zeros_like(sig), np.zeros_like(sig)], axis=1)
    med = median_freq(t, gyro)
    assert 2.0 <= med <= 8.0, f"median {med:.2f} Hz, expected between 2 and 8"
