# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Release step of a sustained sine: level drop from the last 200 ms of the note to 100-200 ms after
note-off (dB, mid channel), for the 262 Hz package sine through the reference and given IRs, and the
same for steady sines at 40 frequencies through each IR (static-IR comparison, reference IR first).
Usage: release_step.py <ir.npy> [...]"""
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
dry = wavfile.read(next(p for p in src.glob('sine*.wav') if 'dry' in p.name))[1].astype(float).mean(axis=1)[:int(17.5 * fs)]
off = int(np.nonzero(np.abs(dry) > 1e-4 * np.abs(dry).max())[0][-1])
db = lambda x: 10 * np.log10(np.mean(x * x) + 1e-30)


def step(y, off):
    m = y.mean(axis=1); return db(m[off - int(.2 * fs):off]) - db(m[off + int(.1 * fs):off + int(.2 * fs)])


def multi(h):
    out = []
    for f in np.geomspace(100, 2000, 40):
        t = np.arange(2 * fs) / fs; x = np.sin(2 * np.pi * f * t) * np.minimum(1, t / .01) * np.minimum(1, (2 - t) / .01)
        y = np.column_stack([fftconvolve(x, h[:, c])[:3 * fs] for c in range(2)]); out.append(step(y, 2 * fs))
    return np.array(out)


print('note-off at %.2f s' % (off / fs))
irs = [('ref IR', ref)] + [(f, np.load(f).astype(float)[:6 * fs]) for f in sys.argv[1:]]
for name, h in irs:
    y = np.column_stack([fftconvolve(dry, h[:, c])[:len(dry) + fs] for c in range(2)])
    m = multi(h)
    print(f'{name:28s} 262Hz step {step(y, off):5.2f} dB | 40 freqs mean {m.mean():5.2f} std {m.std():4.2f} min {m.min():5.2f} max {m.max():5.2f}')
