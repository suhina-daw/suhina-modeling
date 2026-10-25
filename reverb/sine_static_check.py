# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Is the sine's L/R imbalance a property of any static IR? Convolves the dry 262 Hz sine with the
captured reference IR itself (the same IR the fits target) and compares with the captured sine (A).
Also: exact-frequency |H_L|/|H_R| spread over 100-2000 Hz for the reference IR and the candidates."""
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
O = Path(__file__).parent; fs = 48000
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
from metrics import ref
src = CAPTURES / 'samples/pitch'
dry = wavfile.read(next(p for p in src.glob('sine*.wav') if 'dry' in p.name))[1].astype(float).mean(axis=1)
a = wavfile.read(src / 'sine 48k-stereo-20s - 6000ms.wav')[1].astype(float)
n = int(17.5 * fs)
d0 = dry[:n]; F = np.abs(np.fft.rfft(d0)); f0 = np.fft.rfftfreq(n, 1 / fs)[np.argmax(F)]
db = lambda x: 10 * np.log10(np.mean(x * x) + 1e-30)
print('dry sine peak %.2f Hz' % f0)
print('A captured        L-R %+6.2f' % (db(a[:n, 0]) - db(a[:n, 1])))
y = np.column_stack([fftconvolve(dry, ref[:, c])[:len(a)] for c in range(2)])[:n]
print('dry * ref IR      L-R %+6.2f' % (db(y[:, 0]) - db(y[:, 1])))
for name, h in [('ref IR', ref)] + [(f, np.load(f).astype(float)) for f in sys.argv[1:]]:
    h = h[:6 * fs]; k = np.arange(len(h))
    ex = lambda f: np.abs(np.exp(-2j * np.pi * f * k / fs) @ h)
    hl, hr = ex(f0); fr = np.geomspace(100, 2000, 300)
    dd = np.array([20 * np.log10(ex(f)[0] / ex(f)[1]) for f in fr])
    print(f'{name:28s} exact@f0 {20*np.log10(hl/hr):+6.2f}  spread 100-2k std {dd.std():5.2f}  p90|d| {np.percentile(abs(dd),90):5.2f}')
