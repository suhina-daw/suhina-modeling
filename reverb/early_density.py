# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""0-50 ms early density versus the reference: octave-band mid-channel energy in 5 ms frames,
each IR level-matched to the reference over 0-300 ms. Reports mean |error| and the deepest dip
below the reference (dB), averaged over octave bands, frames 2.5-50 ms (predelay excluded).
Import-safe: dips(ir) -> dict; command line: early_density.py <ir.npy> [...]"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
O = Path(__file__).parent; FS = 48000
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
from metrics import ref
from perceptual_objective import OCT

F = 240  # 5 ms


def env(ir):
    x = ir[:int(.3 * FS)]; x = x * np.sqrt(np.sum(ref[:len(x)] ** 2) / np.sum(x * x))
    m = (x[:, 0] + x[:, 1]) / 2
    out = []
    for lo, hi in OCT:
        z = sosfilt(butter(2, [lo, hi], btype='bandpass', fs=FS, output='sos'), m)[:int(.06 * FS)]
        out.append(10 * np.log10(np.mean(z[:len(z) // F * F].reshape(-1, F) ** 2, axis=1) + 1e-30))
    return np.array(out)[:, 1:10]   # 5-50 ms frames (frame 0 holds predelay)


R = env(ref)


def dips(ir):
    e = env(ir) - R
    return dict(mean_abs=float(np.mean(abs(e))), worst=float(np.mean(e.min(axis=1))), per_frame=np.round(e.mean(axis=0), 1).tolist())


if __name__ == '__main__':
    for f in sys.argv[1:]:
        d = dips(np.load(f).astype(float))
        print(f'{f:30s} mean|e| {d["mean_abs"]:5.2f}  worst {d["worst"]:+6.2f}  frames 5..45ms {d["per_frame"]}')


def crest(ir, t1=.1, frame=120):
    """Sparseness of the first t1 s: mean per-frame peak-to-RMS (dB) over 2.5 ms frames, both channels.
    Discrete reflections give a high value, a noise-like (diffused) onset a low one (~11 dB for noise)."""
    x = ir[:int(t1 * FS) // frame * frame]
    f = x.reshape(-1, frame, 2); rms = np.sqrt(np.mean(f * f, axis=1)) + 1e-30
    ok = rms > rms.max() * 1e-3
    pk = np.max(np.abs(f), axis=1) + 1e-30
    return float(np.mean((20 * np.log10(pk / rms))[ok]))


def kurt(ir, t0, t1, frame=480):
    """Peakiness: mean kurtosis of 10 ms frames over t0-t1 s, both channels. 3 = Gaussian noise
    (a dense diffuse tail), higher = sparse discrete reflections. Replaces crest(), which scored the
    smooth decays between discrete spikes as low-peak frames and so ranked a sparse onset as smeared."""
    from scipy.stats import kurtosis
    x = ir[int(t0 * FS):int(t1 * FS)]; n = len(x) // frame
    return float(np.mean([np.mean(kurtosis(x[:n * frame, c].reshape(n, frame), axis=1, fisher=False)) for c in range(2)]))
