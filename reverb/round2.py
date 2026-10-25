# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Round 2 of the parallel research: E combined with Codex's compensated loop scattering,
an interleaved L/R line assignment, and attribute-objective refits; every candidate is
judged by the same four evaluations.

  fresh40   Codex's round 20 stimuli and measure() unchanged (captured-IR predictions)
  sources   Round 13 sources, all held out (envelope + 2-20 Hz modulation)
  objective perceptual_objective.py terms, density, recurrence
  cases     03 bass-upper 1-3 kHz band error and 04 sine per-channel envelope error,
            on the same slices as the listening package

Usage: round2.py <candidate> [--fit]   (writes round2/<candidate>.json)
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve, resample_poly, butter, sosfilt
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
import perceptual_objective as po
from metrics import s, ref, summary, E_EARLY
from fit_refit import LONG
from fit_perceptual import modulation_error
import run_five_rounds as r13

FS = 48000
K = float(np.sqrt(1 / E_EARLY))
E_W = json.loads((DATAFILES / 'e-output-weights.json').read_text())['output_weights']
SCAT = dict(loop_diffusion=0.1, loop_diffusion_scale=1.0, loop_delay_compensation=True)
FEED = dict(variant='early_lr', k=K, ediff=(LONG, .5))
INTER = np.array([0, 1, 0, 1, 0, 1, 0, 1])
CANDIDATES = {
    'r18': ({}, {}),
    'e': (dict(output_weights=E_W), FEED),
    'e_scat01_eweights': (dict(output_weights=E_W, **SCAT), FEED),
    'e_scat01': (dict(output_weights=E_W, **SCAT), FEED),
    'e_inter': (dict(output_weights=E_W), dict(FEED, side=INTER)),
    'e_inter_scat01': (dict(output_weights=E_W, **SCAT), dict(FEED, side=INTER)),
}


def fresh40_data():
    fs = r13.FS; rng = np.random.default_rng(24102419); data = []
    for f in [65.406, 92.499, 138.591, 207.652, 311.127, 466.164, 698.456, 1046.502, 2093.005]:
        for seconds in [.095, .47, .83, 1.7]:
            x = np.zeros(fs * 12); n = round(seconds * fs); w = np.ones(n); w[:48] = np.linspace(0, 1, 48); w[-48:] = np.linspace(1, 0, 48)
            x[:n] = .25 * np.sin(2 * np.pi * f * np.arange(n) / fs) * w; data.append((f'{f}Hz-{seconds}s', x))
    for lo, hi in [(40, 180), (180, 750), (750, 2400), (2400, 5500)]:
        x = np.zeros(fs * 12); x[:fs // 4] = rng.normal(0, .1, fs // 4)
        x = sosfilt(butter(3, [lo, hi], btype='bandpass', fs=fs, output='sos'), x); data.append((f'noise-{lo}-{hi}', x))
    return data


def fresh40(h12, data, refs):
    rows = []
    for (name, x), a in zip(data, refs):
        m = r13.measure(np.column_stack([fftconvolve(x, h12[:, c])[:len(x)] for c in range(2)]), a); m['name'] = name; rows.append(m)
    return rows


def case_metrics(ir):
    out = {}
    root = CAPTURES / 'samples/pitch'
    for label, sound, slot in [('03', 'bass', 3), ('04', 'sine', 0)]:
        dry = r13.read(next(root.glob(sound + ' dry-*.wav'))).mean(axis=1)
        a = r13.read(root / f'{sound} 48k-stereo-20s - 6000ms.wav')
        h = resample_poly(ir, 1, 4, axis=0)
        y = np.column_stack([fftconvolve(dry, h[:, c])[:len(dry)] for c in range(2)])[:len(a)]
        n = 210000; sl = slice(slot * n, (slot + 1) * n); y, a = y[sl], a[sl]
        y = y * np.sqrt(np.sum(a * a) / np.sum(y * y))
        bands = {}
        for lo, hi in [(120, 350), (350, 1000), (1000, 3000), (3000, 5500)]:
            so = butter(3, [lo, hi], btype='bandpass', fs=r13.FS, output='sos')
            bands[f'{lo}-{hi}'] = float(10 * np.log10(np.sum(sosfilt(so, y, axis=0) ** 2) / np.sum(sosfilt(so, a, axis=0) ** 2)))
        e = np.sqrt(np.mean(y[:len(y) // 120 * 120].reshape(-1, 120, 2) ** 2, axis=1)); t = np.sqrt(np.mean(a[:len(a) // 120 * 120].reshape(-1, 120, 2) ** 2, axis=1))
        lr = [float(np.mean(np.abs(20 * np.log10(np.maximum(e[mk, c], 1e-20) / t[mk, c])))) for c in range(2) for mk in [t[:, c] > t[:, c].max() * .001]]
        out[label] = dict(band_error_db=bands, envelope_lr_db=lr)
    return out


def sources(ir):
    data = r13.sources(); h = resample_poly(ir, 1, 4, axis=0); ev = r13.evaluate(h, data); cache = {}
    for c in ev['cases']:
        d = next(dd for dd in data if dd['name'] == c['name']); key = (d['name'].split('-pitch')[0], len(d['x']))
        if key not in cache:
            cache[key] = np.column_stack([fftconvolve(d['x'], h[:, k])[:len(d['x'])] for k in range(2)])
        lo, hi = d['slice']; c['modulation_db'] = modulation_error(cache[key][lo:hi], d['a'])
    g = lambda f, k: float(np.mean([c[k] for c in ev['cases'] if f(c['name'])]))
    o = lambda n: '-pitch' not in n and not n.startswith('vocals'); p = lambda n: '-pitch' in n; v = lambda n: n.startswith('vocals')
    return dict(orig_env=g(o, 'envelope_db'), pitch_env=g(p, 'envelope_db'), vocal_env=g(v, 'envelope_db'),
                orig_mod=g(o, 'modulation_db'), pitch_mod=g(p, 'modulation_db'), vocal_mod=g(v, 'modulation_db'))


def main():
    name = sys.argv[1]; fit = '--fit' in sys.argv
    seed = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--seed=')), None)
    over, kw = CANDIDATES[name]; st = dict(s, **over)
    if seed is not None:
        st['output_weights'] = np.random.default_rng(seed).normal(0, 1, (2, 8)).tolist(); name = f'{name}_seed{seed}'
    out = dict(candidate=name, settings_override={k: v for k, v in over.items() if k != 'output_weights'},
               render_kw={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}, fitted=fit)
    if fit:
        R = po.Reference(ref); x = np.zeros(6 * FS); x[0] = 1.0
        lines, early = model.render(x, st, basis=True, **kw); w0 = np.array(st['output_weights'], float).ravel(); t0 = time.time()
        res = minimize(lambda w: po.objective(lines @ w.reshape(2, 8).T + early, R)[0], w0, method='L-BFGS-B',
                       bounds=[(-8, 8)] * 16, options=dict(maxiter=150))
        st['output_weights'] = res.x.reshape(2, 8).tolist(); out.update(fit_seconds=time.time() - t0, fit_message=str(res.message))
    out['output_weights'] = st.get('output_weights')
    ir = model.render(np.r_[1.0, np.zeros(10 * FS - 1)], st, **kw)
    np.save(WORK / 'round2' / f'{name}-ir.npy', ir.astype(np.float32))
    out['objective'] = po.objective(ir[:6 * FS], po.Reference(ref))
    out['ir'] = summary(ir[:6 * FS])
    data = fresh40_data(); h12 = resample_poly(ir, 1, 4, axis=0)
    ref12 = r13.read(next(CAPTURES.glob('* - 6000ms.wav')))[r13.FS:] / .25
    refs = [np.column_stack([fftconvolve(x, ref12[:, c])[:len(x)] for c in range(2)]) for _, x in data]
    out['fresh40'] = fresh40(h12, data, refs)
    out['sources'] = sources(ir)
    out['cases'] = case_metrics(ir)
    (WORK / 'round2' / f'{name}.json').write_text(json.dumps(out, indent=1, default=float))
    f = out['fresh40']; sc = out['sources']
    print(name, 'fresh40 env %.3f' % np.mean([r['envelope_db'] for r in f]), 'obj %.3f' % out['objective'][0],
          {k: round(v, 3) for k, v in sc.items()}, '03 1-3k %.2f' % out['cases']['03']['band_error_db']['1000-3000'],
          '04 LR', np.round(out['cases']['04']['envelope_lr_db'], 2), flush=True)


if __name__ == '__main__':
    (WORK / 'round2').mkdir(parents=True, exist_ok=True)
    main()
