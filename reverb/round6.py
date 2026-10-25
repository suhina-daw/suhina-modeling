# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Round 6: denser early pattern. The reference's 0-70 ms is a discrete reflection pattern; the
8 taps per channel take its strongest arrivals, and the 10 ms build-up dips (~20, 45 ms) sit where
the reference has further arrivals the 8 taps leave out. Adds up to N extra output-only taps per
channel at the next strongest reference arrivals (peak-picked, 0-70 ms), each with its own fitted
gain, plus an optional shared L<->R crosstalk of the early pattern (the reference shows each
reflection in the other channel at about -17 dB). The tank feed keeps the original 8 taps.
Otherwise as round 5 (shared feed-out and early gain, E structure, early_diffusion).
Usage: round6.py <N extra taps 0-12> <crosstalk 0|1> <seed>"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample_poly, find_peaks, butter, lfilter, fftconvolve
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
import perceptual_objective as po
from metrics import s, ref, summary
import round2
import early_density

po.set_buildup(2.0)
FS = 48000; ED = 0.6; NT = 10 * FS
EDIFF = [[23, 61], [31, 73]]   # native early_diffusion delays per channel


def extra_taps(n, xt):
    """Next strongest reference arrivals per channel (ms, settings time base), excluding the existing taps,
    and with crosstalk also the other channel's taps (their copies are covered by the crosstalk term)."""
    out = []
    for c in range(2):
        x = ref[:int(.07 * FS), c]; a = np.abs(x)
        p, _ = find_peaks(a, height=a.max() * .05, distance=12)
        have = np.array(s['early_tap_times_ms'][c] + (s['early_tap_times_ms'][1 - c] if xt else [])) + 2.5
        p = [i for i in p if np.min(abs(i / FS * 1000 - have)) > .3]
        p = sorted(p, key=lambda i: -a[i])[:n]
        out.append(sorted(round(float(i / FS * 1000 - 2.5), 4) for i in p))
    return out


def tap_basis(times, st):
    pre = int(np.floor(st.get('predelay_ms', 2.5) * FS / 1000 + .5))
    hb, ha = butter(1, st.get('low_cut_hz', 120), 'highpass', fs=FS)
    lb, la = butter(1, st.get('high_cut_hz', 14000), 'lowpass', fs=FS)
    cols = []
    for c in range(2):
        for ms in times[c]:
            z = np.zeros(NT); z[pre + int(np.floor(ms * FS / 1000 + .5))] = 1.0
            for d in (EDIFF[c] if ED > 0 else []):   # an allpass with g 0 is a pure delay, so skip it
                z = model.allpass(z, d, ED)
            col = np.zeros((NT, 2)); col[:, c] = lfilter(lb, la, lfilter(hb, ha, z)); cols.append(col)
    return cols


def main():
    n, xt, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    over, kw = round2.CANDIDATES['e']; st = dict(s, **over, early_diffusion=ED)
    R = po.Reference(ref); x = np.zeros(NT); x[0] = 1.0
    lines, early, fed = model.render(x, st, basis=True, basis_feed=True, **kw)
    times = extra_taps(n, xt); cols = tap_basis(times, st); k = len(cols)
    w0 = np.array(st['output_weights'], float).ravel() if seed == 0 else np.random.default_rng(100 + seed).normal(0, 1, 16)
    g0 = [float(ref[int((t + 2.5) * FS / 1000 + .5), c]) / .05 for c in range(2) for t in times[c]]
    w0 = np.r_[w0, 0.5, 1.0, 0.0, g0]   # shared feed-out, shared early gain, crosstalk, extra tap gains
    six = slice(0, 6 * FS)
    L6, E6, F6, X6 = lines[six], early[six], fed[six], early[six][:, ::-1]
    C6 = np.stack([c[six] for c in cols], axis=-1) if k else None

    def ir_of(w, L=L6, E=E6, F=F6, X=X6, C=C6):
        y = L @ w[:16].reshape(2, 8).T + E * w[17] + F * w[16] + X * w[18]
        return y + (C @ w[19:] if k else 0)
    bounds = [(-8, 8)] * 16 + [(-4, 4)] * 2 + [(-1, 1) if xt else (0, 0)] + [(-4, 4)] * k
    t0 = time.time()
    res = minimize(lambda w: po.objective(ir_of(w), R)[0], w0, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=200))
    w = res.x
    ir = ir_of(w, lines, early, fed, early[:, ::-1], np.stack(cols, axis=-1) if k else None)
    assert st.get('width', 1.0) == 1.0  # the basis path applies no width
    tag = f'n{n}_x{xt}_s{seed}'
    np.save(WORK / 'round6' / f'{tag}-ir.npy', ir.astype(np.float32))
    gi = iter(w[19:])
    out = dict(candidate=tag, early_diffusion=ED, feed_out=float(w[16]), early_gain=float(w[17]), crosstalk=float(w[18]),
               extra_taps=[[[t, float(next(gi))] for t in times[c]] for c in range(2)], output_weights=w[:16].reshape(2, 8).tolist(),
               fit_seconds=time.time() - t0, fit_message=str(res.message),
               objective=po.objective(ir[:6 * FS], R), ir=summary(ir[:6 * FS]), early_density=early_density.dips(ir))
    data = round2.fresh40_data()
    ref12 = round2.r13.read(next(CAPTURES.glob('* - 6000ms.wav')))[round2.r13.FS:] / .25
    refs = [np.column_stack([fftconvolve(xx, ref12[:, c])[:len(xx)] for c in range(2)]) for _, xx in data]
    out['fresh40'] = round2.fresh40(resample_poly(ir, 1, 4, axis=0), data, refs)
    out['sources'] = round2.sources(ir)
    out['cases'] = round2.case_metrics(ir)
    (WORK / 'round6' / f'{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    print(tag, 'obj %.3f' % out['objective'][0], {kk: round(v, 3) for kk, v in out['objective'][1].items()},
          'fresh40 %.3f' % np.mean([r['envelope_db'] for r in out['fresh40']]),
          {kk: round(v, 3) for kk, v in out['sources'].items()}, 'early0-50 %.2f worst %.2f' % (out['early_density']['mean_abs'], out['early_density']['worst']),
          'xt %.3f' % w[18], flush=True)


if __name__ == '__main__':
    (WORK / 'round6').mkdir(parents=True, exist_ok=True)
    main()
