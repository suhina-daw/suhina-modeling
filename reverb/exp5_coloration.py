# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment 5: does the early FIR pattern colour the tail when it feeds the tank?

Coloration = log-magnitude deviation of the late IR (0.2-2 s) from the reference, smoothed to
1/12 octave, 60-8000 Hz, after removing the mean offset. Also the early FIR's own
magnitude ripple. Adds an allpass-only stereo feed (spectrally flat) as the control.
"""
import json, sys
from pathlib import Path
import numpy as np
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
from metrics import s, ref, E_EARLY

FS = 48000
x = np.zeros(6 * FS); x[0] = 1.0
f = np.fft.rfftfreq(2 * FS * 2, 1 / FS)
centres = 60 * 2 ** (np.arange(0, np.log2(8000 / 60) * 12) / 12)


def smooth_db(sig):
    p = np.sum(np.abs(np.fft.rfft(sig, n=4 * FS, axis=0)) ** 2, axis=1)
    out = []
    for c in centres:
        m = (f >= c * 2 ** (-1 / 24)) & (f < c * 2 ** (1 / 24))
        out.append(10 * np.log10(np.mean(p[m]) + 1e-30))
    return np.array(out)


def coloration(ir):
    a = smooth_db(ir[int(.2 * FS):2 * FS]); b = smooth_db(ref[int(.2 * FS):2 * FS])
    d = a - b
    return float(np.std(d - d.mean())), d - d.mean()


def ripple(sig):
    d = smooth_db(np.column_stack([sig, sig]) if sig.ndim == 1 else sig)
    return float(np.std(d - d.mean()))


if __name__ == '__main__':
    early = model.early_taps(np.r_[1.0, np.zeros(FS - 1)], s)
    print('early FIR ripple L/R (1/12 oct std dB):', ripple(early[:, :1].repeat(2, 1)), ripple(early[:, 1:].repeat(2, 1)))
    LONG = ([443, 691, 1009, 1361], [479, 727, 1061, 1427])
    k1 = float(np.sqrt(1 / E_EARLY))
    res = {}
    for name, kw in [('base', {}), ('early_lr_r1', dict(variant='early_lr', k=k1)),
                     ('long_g0.5_lr_r1', dict(variant='early_lr', k=k1, ediff=(LONG, .5)))]:
        ir = model.render(x, s, **kw)
        res[name], _ = coloration(ir)
        print(name, 'late coloration vs ref, std dB:', round(res[name], 3))
    (WORK / 'exp5-coloration.json').write_text(json.dumps(res, indent=1))
