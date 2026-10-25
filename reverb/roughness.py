# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sine-tail roughness: how much a sustained tone's decay beats, as a statistic over many
frequencies (an attribute, not per-frequency matching). For each frequency, a 1.0 s tone
is convolved with the IR; after it stops, the log envelope (20 ms RMS) is detrended with
a straight line over 0.2-3 s and its standard deviation is the roughness in dB."""
import numpy as np
from scipy.signal import fftconvolve, resample_poly
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE

FREQS = 55 * 2 ** (np.arange(0, 40) / 7)     # 55 Hz .. ~2.8 kHz, 40 log-spaced


def roughness(ir, fs=48000):
    h = resample_poly(ir, 1, 4, axis=0); f12 = fs // 4
    n = f12; t = np.arange(n) / f12; win = np.minimum(1, np.minimum(t, 1 - t) / .01)
    out = np.zeros((len(FREQS), 2))
    for i, f in enumerate(FREQS):
        x = np.sin(2 * np.pi * f * t) * win
        y = np.column_stack([fftconvolve(x, h[:, c])[:n + 4 * f12] for c in range(2)])[n:]
        w = 240; m = len(y) // w * w
        e = 10 * np.log10(np.mean(y[:m].reshape(-1, w, 2) ** 2, axis=1) + 1e-30)
        tt = np.arange(len(e)) * w / f12; sel = (tt > .2) & (tt < 3)
        for c in range(2):
            p = np.polyfit(tt[sel], e[sel, c], 1)
            out[i, c] = np.std(e[sel, c] - np.polyval(p, tt[sel]))
    return out
