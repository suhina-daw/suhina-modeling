# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fair structural comparison: refit only the 2x8 late output weights of each topology with
the same optimiser and budget, on a training split, and report a held-out split.

Train: 6 original non-vocal sources + pitch segments 1-3 of each of 5 waveforms (21 cases).
Test: pitch segments 4-5 (10 cases) + fresh vocals (2 cases). Never used for fitting.
Objective: Codex's Round 13 score (measure(): per-case energy normalisation), equal weight
for originals and pitch, on train only. Usage: fit_refit.py <candidate-name> [maxfev]
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
from metrics import s, E_EARLY, summary
from exp5_coloration import coloration
import run_five_rounds as r13

FS = 48000
MID = ([241, 379, 563, 797], [263, 409, 601, 853])
LONG = ([443, 691, 1009, 1361], [479, 727, 1061, 1427])
k1 = float(np.sqrt(1 / E_EARLY))
CANDIDATES = {
    'base': {},
    'early_lr_r1': dict(variant='early_lr', k=k1),
    'firdiff_long_g0.5_lr_r1': dict(variant='early_lr', k=k1, ediff=(LONG, .5)),
    'ap_mid_g0.5_k1_keep1': dict(variant='ap_lr', k=1, mono_keep=1, ediff=(MID, .5)),
    'ap_mid_g0.7_k1_keep1': dict(variant='ap_lr', k=1, mono_keep=1, ediff=(MID, .7)),
    'ap_long_g0.5_k1_keep1': dict(variant='ap_lr', k=1, mono_keep=1, ediff=(LONG, .5)),
}


def split(name):
    if '-pitch' in name:
        return 'train' if int(name[-1]) <= 3 else 'test'
    return 'test' if name.startswith('vocals') else 'train'


def main():
    cname = sys.argv[1]
    maxfev = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    kw = CANDIDATES[cname]
    x = np.zeros(10 * FS); x[0] = 1.0
    lines, early = model.render(x, s, basis=True, **kw)
    bl = resample_poly(lines, 1, 4, axis=0); be = resample_poly(early, 1, 4, axis=0)
    data = r13.sources()
    conv = {}
    for d in data:
        key = (d['name'].split('-pitch')[0], len(d['x']))
        if key not in conv:
            conv[key] = (np.column_stack([fftconvolve(d['x'], bl[:, i])[:len(d['x'])] for i in range(8)]).astype(np.float32),
                         np.column_stack([fftconvolve(d['x'], be[:, c])[:len(d['x'])] for c in range(2)]).astype(np.float32))
    cases = []
    for d in data:
        key = (d['name'].split('-pitch')[0], len(d['x']))
        lo, hi = d['slice']
        cases.append(dict(name=d['name'], part=split(d['name']), a=d['a'], L=conv[key][0][lo:hi], E=conv[key][1][lo:hi]))

    def evaluate(w, part):
        W = w.reshape(2, 8)
        rows = []
        for c in cases:
            if part != 'all' and c['part'] != part:
                continue
            y = (c['L'] @ W.T.astype(np.float32) + c['E']).astype(float)
            m = r13.measure(y, c['a']); m['name'] = c['name']; m['part'] = c['part']; rows.append(m)
        return rows

    def score(rows):
        orig = [r['score'] for r in rows if '-pitch' not in r['name'] and not r['name'].startswith('vocals')]
        pitch = [r['score'] for r in rows if '-pitch' in r['name']]
        return .5 * np.mean(orig) + .5 * np.mean(pitch) if orig else float(np.mean(pitch))

    def agg(rows):
        g = lambda f: float(np.mean([r['envelope_db'] for r in rows if f(r['name'])])) if any(f(r['name']) for r in rows) else None
        return dict(orig=g(lambda n: '-pitch' not in n and not n.startswith('vocals')), pitch=g(lambda n: '-pitch' in n),
                    vocal=g(lambda n: n.startswith('vocals')), test_pitch_score=float(np.mean([r['score'] for r in rows if '-pitch' in r['name'] and r['part'] == 'test'])),
                    train_score=float(score([r for r in rows if r['part'] == 'train'])))

    w0 = np.array(s['output_weights'], dtype=float).ravel()
    before = evaluate(w0, 'all')
    t = time.time(); n = [0]

    def objective(w):
        n[0] += 1
        return score(evaluate(w, 'train'))
    res = minimize(objective, w0, method='Powell', options=dict(maxfev=maxfev, xtol=1e-3, ftol=1e-5))
    after = evaluate(res.x, 'all')
    ow = res.x.reshape(2, 8)
    ir = model.render(x, {**s, 'output_weights': ow.tolist()}, **kw)[:6 * FS]
    out = dict(candidate=cname, settings_kw={k: (v if not isinstance(v, tuple) else [v[0], v[1]]) for k, v in kw.items()},
               evals=n[0], seconds=time.time() - t, before=agg(before), after=agg(after),
               ir_after=summary(ir), coloration_after=coloration(ir)[0], output_weights=ow.tolist(), cases_after=after)
    (WORK / 'refit').mkdir(parents=True, exist_ok=True)
    (WORK / 'refit' / f'{cname}.json').write_text(json.dumps(out, indent=1, default=float))
    print(cname, json.dumps({k: out[k] for k in ['evals', 'before', 'after', 'coloration_after']}, default=float))


if __name__ == '__main__':
    main()
