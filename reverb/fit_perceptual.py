# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fit the 2x8 late output weights of one topology to the reference IR with the
attribute-level objective in perceptual_objective.py (L-BFGS-B, numerical gradient).

No source recordings are used in fitting, so every source case is held out. The holdout
report uses Codex's Round 13 measure() unchanged plus a 2-20 Hz modulation error.
Usage: fit_perceptual.py <candidate> [maxiter]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve, resample_poly
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
import perceptual_objective as po
from metrics import s, ref, summary
from fit_refit import CANDIDATES
import run_five_rounds as r13

FS = 48000


def modulation_error(y, a):
    """|dB| difference of 2-20 Hz envelope modulation power in four octave bands (12 kHz input)."""
    fs = 12000
    out = []
    for lo, hi in [(90, 180), (355, 710), (1400, 2800), (2800, 5600)]:
        def mod(z):
            n = len(z) // 120 * 120
            spec = np.abs(np.fft.rfft(z[:n].sum(axis=1).reshape(-1, 120) * np.hanning(120), n=512, axis=1)) ** 2
            f = np.fft.rfftfreq(512, 1 / fs)
            env = 10 * np.log10(spec[:, (f >= lo) & (f < hi)].sum(axis=1) + 1e-20)
            env = env[env > env.max() - 50] if np.any(env > env.max() - 50) else env
            E = np.abs(np.fft.rfft(env - env.mean())) ** 2
            fm = np.fft.rfftfreq(len(env), 1 / 100)
            return np.sum(E[(fm >= 2) & (fm <= 20)]) / max(len(env), 1)
        out.append(abs(10 * np.log10(max(mod(y), 1e-20) / max(mod(a), 1e-20))))
    return float(np.mean(out))


def holdout(ir):
    data = r13.sources()
    h = resample_poly(ir, 1, 4, axis=0)
    ev = r13.evaluate(h, data)
    cache = {}
    for c in ev['cases']:
        d = next(dd for dd in data if dd['name'] == c['name'])
        key = (d['name'].split('-pitch')[0], len(d['x']))
        if key not in cache:
            cache[key] = np.column_stack([fftconvolve(d['x'], h[:, k])[:len(d['x'])] for k in range(2)])
        lo, hi = d['slice']
        c['modulation_db'] = modulation_error(cache[key][lo:hi], d['a'])
    g = lambda f, k: float(np.mean([c[k] for c in ev['cases'] if f(c['name'])]))
    is_o = lambda n: '-pitch' not in n and not n.startswith('vocals')
    is_p = lambda n: '-pitch' in n
    is_v = lambda n: n.startswith('vocals')
    return dict(orig_env=g(is_o, 'envelope_db'), pitch_env=g(is_p, 'envelope_db'), vocal_env=g(is_v, 'envelope_db'),
                orig_mod=g(is_o, 'modulation_db'), pitch_mod=g(is_p, 'modulation_db'), vocal_mod=g(is_v, 'modulation_db'),
                score=ev['score'], cases=ev['cases'])


def main():
    cname = sys.argv[1]
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    kw = CANDIDATES[cname]
    R = po.Reference(ref)
    x = np.zeros(6 * FS); x[0] = 1.0
    lines, early = model.render(x, s, basis=True, **kw)
    w0 = np.array(s['output_weights'], float).ravel()
    log = open(WORK / 'perceptual' / f'{cname}.progress.txt', 'w')
    t0 = time.time(); n = [0]

    def f(w):
        n[0] += 1
        v, _ = po.objective(lines @ w.reshape(2, 8).T + early, R)
        if n[0] % 17 == 0:
            log.write(f'{n[0]} {time.time() - t0:.0f}s {v:.4f}\n'); log.flush()
        return v
    before = po.objective(lines @ w0.reshape(2, 8).T + early, R)
    res = minimize(f, w0, method='L-BFGS-B', bounds=[(-8, 8)] * 16, options=dict(maxiter=maxiter))
    W = res.x.reshape(2, 8)
    after = po.objective(lines @ W.T + early, R)
    full = model.render(np.r_[1.0, np.zeros(10 * FS - 1)], {**s, 'output_weights': W.tolist()}, **kw)
    out = dict(candidate=cname, evals=n[0], seconds=time.time() - t0, message=str(res.message),
               objective_before=before, objective_after=after, output_weights=W.tolist(),
               ir_after=summary(full[:6 * FS]), holdout_after=holdout(full))
    (WORK / 'perceptual' / f'{cname}.json').write_text(json.dumps(out, indent=1, default=float))
    h = out['holdout_after']
    print(cname, f"obj {before[0]:.3f}->{after[0]:.3f}", {k: round(v, 4) for k, v in after[1].items()},
          {k: round(v, 3) for k, v in h.items() if k != 'cases'})


if __name__ == '__main__':
    (WORK / 'perceptual').mkdir(parents=True, exist_ok=True)
    main()
