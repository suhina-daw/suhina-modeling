# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""L/R balance of the 262 Hz sine (pitch slot 0) for given IR files versus the reference capture.
Prints whole-slot L-R level difference and the same over the tail (after the dry note ends) in 1 s windows.
Usage: sine_balance.py <ir.npy> [...]"""
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
O = Path(__file__).parent; fs = 48000
src = CAPTURES / 'samples/pitch'
dry = wavfile.read(next(p for p in src.glob('sine*.wav') if 'dry' in p.name))[1].astype(float).mean(axis=1)
a = wavfile.read(src / 'sine 48k-stereo-20s - 6000ms.wav')[1].astype(float)
n = int(17.5 * fs); sl = slice(0, n)


def report(name, y):
    db = lambda x: 10 * np.log10(np.mean(x * x) + 1e-30)
    win = [db(y[i * fs:(i + 1) * fs, 0]) - db(y[i * fs:(i + 1) * fs, 1]) for i in range(int(len(y) / fs))]
    print(f'{name:28s} L-R {db(y[:, 0]) - db(y[:, 1]):+6.2f} dB | per s:', ' '.join(f'{v:+5.1f}' for v in win[:14]))


report('A ref', a[sl])
for f in sys.argv[1:]:
    h = np.load(f).astype(float)
    y = np.column_stack([fftconvolve(dry, h[:, c])[:len(a)] for c in range(2)])[sl]
    report(Path(f).parent.name + '/' + Path(f).stem, y)
