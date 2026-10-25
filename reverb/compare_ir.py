# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""REW-style comparison of one IR against the reference, per channel, on a common time axis
(input impulse = 0), level-matched over 0-6 s: impulse (dB, 0-1 s), ETC 0-150 ms, step response,
Schroeder decay (EDC) and 1/6-octave smoothed magnitude. Writes compare-<label>.png.
Usage: compare_ir.py <label> <ir.npy>"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
O = Path(__file__).parent; FS = 48000
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
from metrics import ref
label, path = sys.argv[1], sys.argv[2]
h = np.load(path).astype(float)[:6 * FS]; h *= np.sqrt(np.sum(ref ** 2) / np.sum(h ** 2))
db = lambda x: 20 * np.log10(np.abs(x) + 1e-12)
fig, ax = plt.subplots(5, 2, figsize=(18, 20))
for c in range(2):
    r, y = ref[:, c], h[:, c]
    t = np.arange(FS) / FS
    a = ax[0, c]; a.plot(t, db(r[:FS]), lw=.3, label='A ref'); a.plot(t, db(y[:FS]), lw=.3, alpha=.7, label=label)
    a.set_ylim(-70, -10); a.set_title(f'Impulse dB 0-1 s, {"LR"[c]}'); a.legend()
    n = int(.15 * FS); t = np.arange(n) / FS * 1000
    a = ax[1, c]; a.plot(t, db(hilbert(r[:n])), lw=.6, label='A ref'); a.plot(t, db(hilbert(y[:n])), lw=.6, alpha=.7, label=label)
    a.set_ylim(-80, -15); a.set_xticks(range(0, 151, 10)); a.set_title(f'ETC 0-150 ms, {"LR"[c]}'); a.legend(); a.grid(alpha=.3)
    t = np.arange(int(.5 * FS)) / FS * 1000
    a = ax[2, c]; a.plot(t, np.cumsum(r[:len(t)]), lw=.6, label='A ref'); a.plot(t, np.cumsum(y[:len(t)]), lw=.6, alpha=.8, label=label)
    a.set_title(f'Step response 0-500 ms, {"LR"[c]}'); a.legend(); a.grid(alpha=.3)
    t = np.arange(len(r)) / FS
    e = lambda x: 10 * np.log10(np.cumsum(x[::-1] ** 2)[::-1] / np.sum(x ** 2) + 1e-30)
    a = ax[3, c]; a.plot(t, e(r), label='A ref'); a.plot(t, e(y), label=label); a.set_ylim(-60, 0)
    a.set_title(f'Schroeder decay (EDC), {"LR"[c]}'); a.legend(); a.grid(alpha=.3)
    f = np.fft.rfftfreq(len(r), 1 / FS)
    def sm(x):
        P = np.abs(np.fft.rfft(x)) ** 2; cs = np.r_[0, np.cumsum(P)]; fc = np.geomspace(30, 20000, 600)
        lo = np.searchsorted(f, fc * 2 ** (-1 / 12)); hi = np.maximum(np.searchsorted(f, fc * 2 ** (1 / 12)), lo + 1)
        return fc, 10 * np.log10((cs[hi] - cs[lo]) / (hi - lo))
    a = ax[4, c]; fc, pr = sm(r); _, py = sm(y); a.semilogx(fc, pr, label='A ref'); a.semilogx(fc, py, label=label)
    a.set_title(f'Magnitude 1/6 oct, {"LR"[c]}'); a.legend(); a.grid(alpha=.3, which='both')
fig.tight_layout(); out = WORK / f'compare-{label}.png'; fig.savefig(out, dpi=70); print(out)
