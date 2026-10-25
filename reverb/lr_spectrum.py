# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Narrow-band L/R level difference of IRs: |H_L|/|H_R| smoothed over 1/N octave, 100 Hz - 8 kHz.
Prints the value at 262 Hz and the spread (std, 90th percentile of |L-R|) per IR, reference first.
Usage: lr_spectrum.py <N> <ir.npy> [...]"""
import sys
from pathlib import Path
import numpy as np
O = Path(__file__).parent; fs = 48000
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
from metrics import ref


def lr(h, N):
    h = h[:6 * fs]; H = np.abs(np.fft.rfft(h, axis=0)) ** 2; f = np.fft.rfftfreq(len(h), 1 / fs)
    c = np.cumsum(H, axis=0)
    fc = np.geomspace(100, 8000, 400); lo = np.searchsorted(f, fc * 2 ** (-.5 / N)); hi = np.searchsorted(f, fc * 2 ** (.5 / N))
    e = (c[hi] - c[lo]); d = 10 * np.log10(e[:, 0] / e[:, 1])
    return fc, d


N = float(sys.argv[1])
for name, h in [('ref', ref)] + [(f, np.load(f).astype(float)) for f in sys.argv[2:]]:
    fc, d = lr(h, N); i = np.argmin(abs(fc - 262))
    print(f'{name:28s} @262 {d[i]:+6.2f}  mean {d.mean():+5.2f}  std {d.std():5.2f}  p90|d| {np.percentile(abs(d), 90):5.2f}')
