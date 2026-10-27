# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Signal measurements; no capture paths or filesystem writes on import."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy import signal


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    sr, x = wavfile.read(path)
    if x.ndim != 2 or x.shape[1] != 2 or sr != 48000:
        raise ValueError(f'{path}: expected stereo 48000 Hz')
    if np.issubdtype(x.dtype, np.signedinteger):
        x = x.astype(np.float64) / (2 ** (8 * x.dtype.itemsize - 1))
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128) / 128
    else:
        x = x.astype(np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f'{path}: empty or non-finite audio')
    return sr, x


def write(path, sr, x):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not np.isfinite(x).all():
        raise ValueError('non-finite output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        wavfile.write(stream, sr, np.asarray(x, dtype=np.float32))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else 0.0


def db(x):
    return float(20 * np.log10(max(float(x), 1e-15)))


def active_mask(x, sr=48000):
    """10 ms source-energy windows, fixed absolute floor; includes zero crossings."""
    energy = np.mean(x * x, axis=1)
    w = max(1, round(sr * .01))
    return signal.convolve(energy, np.ones(w) / w, mode='same') > 1e-10


def delay(reference, candidate, limit=128):
    """Candidate lag in samples, parabolic fractional refinement, no gain fitting."""
    a, b = reference[:, 0], candidate[:, 0]
    c = signal.correlate(b, a, mode='full', method='fft')
    lags = signal.correlation_lags(len(b), len(a))
    valid = np.flatnonzero(np.abs(lags) <= limit)
    i = int(valid[np.argmax(c[valid])])
    offset = 0.0
    if 0 < i < len(c) - 1:
        denominator = c[i-1] - 2*c[i] + c[i+1]
        if abs(denominator) > 1e-20:
            offset = float(.5 * (c[i-1] - c[i+1]) / denominator)
    return float(lags[i]) + offset


def advance(x, samples):
    """Windowed-sinc fractional advance, zero padded (never wraps the end)."""
    integer = int(np.floor(samples))
    fractional = samples - integer
    taps = np.arange(-48, 49)
    kernel = np.sinc(taps + fractional) * np.kaiser(len(taps), 10)
    kernel /= kernel.sum()
    filtered = signal.convolve(x, kernel[:, None], mode='same', method='fft')
    result = np.zeros_like(x)
    lo, hi = max(0, -integer), min(len(x), len(x) - integer)
    if hi > lo:
        result[lo:hi] = filtered[lo+integer:hi+integer]
    return result


def metrics(reference, candidate, mask):
    if not mask.any():
        return {'status': 'silent-input'}
    a, b = reference[mask], candidate[mask]
    return {'status': 'measured', 'active_frames': int(mask.sum()),
            'reference_rms_dbfs': db(rms(a)), 'candidate_rms_dbfs': db(rms(b)),
            'gain_error_db': db(rms(b)) - db(rms(a)),
            'error_relative_db': db(rms(b-a)) - db(rms(a)),
            'reference_peak': float(np.max(np.abs(a))),
            'candidate_peak': float(np.max(np.abs(b)))}


def harmonics(x, sr, frequency, count=10):
    """Least-squares sinusoidal amplitudes; supplied known fundamental required."""
    if not np.isfinite(frequency) or not 0 < frequency < sr/2 or count < 1:
        raise ValueError('fundamental must be finite and below Nyquist; count must be positive')
    y = x[:, 0] if x.ndim == 2 else x
    if len(y) < 2 or not np.isfinite(y).all():
        raise ValueError('harmonics require finite nonempty audio')
    t = np.arange(len(y)) / sr
    orders = np.arange(1, min(count, int((sr/2 - 1) / frequency)) + 1)
    cols = [np.ones(len(y))]
    for k in orders:
        cols.extend([np.cos(2*np.pi*k*frequency*t), np.sin(2*np.pi*k*frequency*t)])
    fit = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    amplitudes = np.hypot(fit[1::2], fit[2::2])
    # A frequency doubler can suppress the fundamental. A huge ratio to numerical
    # leakage is not a useful THD measurement; retain the absolute amplitudes.
    measurable = amplitudes[0] > max(1e-12, np.linalg.norm(amplitudes)*1e-5)
    return {'frequency_hz': float(frequency), 'amplitudes': amplitudes.tolist(),
            'thd_status': 'measured' if measurable else 'fundamental-below-floor',
            'thd': float(np.linalg.norm(amplitudes[1:]) / amplitudes[0]) if measurable else None}
