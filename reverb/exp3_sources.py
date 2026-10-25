# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment 3: source-level comparison of early->tank candidates.

Uses Codex's Round 13 measure()/evaluate() unchanged for comparability (per-case energy
normalisation, stereo-summed envelope), and adds per-channel L/R envelope error.
The network is linear and time-invariant, and model.py matches native to 1e-7, so
convolving the dry source with a candidate's model IR equals its native output.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve, resample_poly
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
from metrics import s, E_EARLY, summary
import run_five_rounds as r13

FS = 48000
N = 10 * FS
x = np.zeros(N); x[0] = 1.0
LONG = ([443, 691, 1009, 1361], [479, 727, 1061, 1427])
MID = ([241, 379, 563, 797], [263, 409, 601, 853])


def late_energy(ir):
    return np.sum(ir[int(.05 * FS):] ** 2)


base_ir = model.render(x, s)
E_BASE = late_energy(base_ir)


def candidate(name, **kw):
    k0 = kw.pop('k0', 0.0); keep0 = kw.pop('keep', 1.0); preserve = kw.pop('preserve', True)
    ir = model.render(x, s, k=k0, mono_keep=keep0, **kw)
    if preserve:
        c = float(np.sqrt(E_BASE / late_energy(ir)))
        ir = model.render(x, s, k=k0 * c, mono_keep=keep0 * c, **kw)
    return name, ir


k1 = float(np.sqrt(1 / E_EARLY))
CANDS = [
    ('base', base_ir),
    candidate('early_lr_r1', variant='early_lr', k0=k1),
    candidate('early_mono_r1', variant='early_mono', k0=k1),
    candidate('long_g0.5_lr_r1', variant='early_lr', k0=k1, ediff=(LONG, .5)),
    candidate('long_g0.5_mono_r1', variant='early_mono', k0=k1, ediff=(LONG, .5)),
    candidate('mid_g0.5_lr_r1', variant='early_lr', k0=k1, ediff=(MID, .5)),
    candidate('long_g0.5_lr_r1_raw', variant='early_lr', k0=k1, ediff=(LONG, .5), preserve=False),
]


def env_lr(y):
    n = len(y) // 120 * 120
    return np.sqrt(np.mean(y[:n].reshape(-1, 120, 2) ** 2, axis=1))


def lr_error(y, a):
    y = y[:len(a)] * np.sqrt(np.sum(a * a) / max(np.sum(y[:len(a)] ** 2), 1e-30))
    e, t = env_lr(y), env_lr(a)
    out = []
    for ch in range(2):
        m = t[:, ch] > t[:, ch].max() * .001
        out.append(float(np.mean(np.abs(20 * np.log10(np.maximum(e[m, ch], 1e-20) / t[m, ch])))))
    return out


data = r13.sources()
results = {}
for name, ir in CANDS:
    h = resample_poly(ir, 1, 4, axis=0)
    ev = r13.evaluate(h, data)
    cache = {}
    for case in ev['cases']:
        d = next(dd for dd in data if dd['name'] == case['name'])
        key = (d['name'].split('-pitch')[0], len(d['x']))
        if key not in cache:
            cache[key] = np.column_stack([fftconvolve(d['x'], h[:, c])[:len(d['x'])] for c in range(2)])
        lo, hi = d['slice']
        case['envelope_lr_db'] = lr_error(cache[key][lo:hi], d['a'])
    ir_sum = summary(ir[:6 * FS])
    results[name] = dict(ir=ir_sum, score=ev['score'], original_envelope=ev['original_envelope'],
                         pitch_envelope=ev['pitch_envelope'], vocal_envelope=ev['vocal_envelope'], cases=ev['cases'])
    lr = np.mean([c['envelope_lr_db'] for c in ev['cases']], axis=0)
    print(f"{name:24s} score {ev['score']:.4f} orig {ev['original_envelope']:.4f} pitch {ev['pitch_envelope']:.4f} "
          f"vocal {ev['vocal_envelope']:.4f} LR {lr.round(4)} dens {ir_sum['density_mae_50_700ms']:.3f} "
          f"rec {ir_sum['recurrence_max_abs']:.3f} e0-50 {ir_sum['segment_energy_vs_ref_db']['0-0.05']}", flush=True)
(WORK / 'exp3-results.json').write_text(json.dumps(results, indent=1, default=float))
np.savez_compressed(WORK / 'exp3-irs.npz', **{n: ir for n, ir in CANDS})
