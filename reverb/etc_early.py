# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""ETC (|analytic signal|, dB) of the first 100 ms per channel on a common time axis (input impulse = 0),
each IR level-matched to the reference over 0-6 s. Writes etc-early.png.
Usage: etc_early.py <label=ir.npy> [...]"""
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
items = [('A ref', ref)] + [(a.split('=')[0], np.load(a.split('=')[1]).astype(float)[:6 * FS]) for a in sys.argv[1:]]
n = int(.1 * FS); t = np.arange(n) / FS * 1000
fig, ax = plt.subplots(len(items), 2, figsize=(16, 2.3 * len(items)), sharex=True, sharey=True)
for i, (name, h) in enumerate(items):
    h = h * np.sqrt(np.sum(ref ** 2) / np.sum(h ** 2))
    for c in range(2):
        e = 20 * np.log10(np.abs(hilbert(h[:n, c])) + 1e-9)
        ax[i, c].plot(t, e, lw=.5); ax[i, c].set_ylim(-80, -15); ax[i, c].grid(alpha=.3)
        ax[i, c].set_title(f'{name} {"LR"[c]}', fontsize=9)
for a in ax[-1]:
    a.set_xlabel('ms'); a.set_xticks(range(0, 101, 5))
fig.tight_layout(); fig.savefig(WORK / 'etc-early.png', dpi=80)
