# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Round 7: discrete early instead of smeared. The reference's first 100 ms is a train of clean,
separate reflections that grows denser (etc-early.png); ours turns each reflection into a noise
burst through three spreaders: early_diffusion on the output copy, the allpasses on the tank feed
(and its direct output), and the input diffusion before the tank. This round switches them off
one by one, optionally adds the extra discrete taps of round 6, and fits as round 5 (shared gains).
Usage: round7.py <input diffusion> <early_diffusion> <feed: allpass|allpass_noout|ap<g>_noout|raw|none> <extra taps> <seed> [crosstalk 0|1] [burst weight] [build-up window s] [loop scatter g] [iwflat 0|1]"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample_poly, fftconvolve
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
import perceptual_objective as po
from metrics import s, ref, summary
import round2, round6
import early_density

po.set_buildup(2.0)
FS = 48000; NT = 10 * FS
KW = [(.0025, .05), (.05, .15), (.3, 1)]   # kurtosis windows: onset, build-up, tail


def main():
    dg, ed, feed, n, seed = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    xt = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    bw = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    po.set_burst(bw)
    bu = float(sys.argv[8]) if len(sys.argv) > 8 else 0.3
    po.set_buildup(2.0, bu)
    sg = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    iwf = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    over, kw = round2.CANDIDATES['e']; kw = dict(kw)
    st = dict(s, **over, early_diffusion=ed, diffusion=dg)
    if iwf:
        # equal-magnitude input weights (signs kept): round 18's 1.6-2.0 on lines 4-7 make line 4's
        # concentrated second-order arrival at ~105 ms dominate the tank's first 150 ms
        st['input_weights'] = [float(np.sign(v)) for v in st['input_weights']]
    if sg:
        # Codex's delay-compensated loop scattering (round2.SCAT with its own g): spreads recirculations
        st.update(round2.SCAT, loop_diffusion=sg)
    if feed.startswith('ap') and feed.endswith('_noout'):
        # feed allpass chain with its own coefficient: at g .5 the chain's strongest echo sits at the sum of
        # its delays (73/77 ms, (1-g^2)^4 = .32) and returns through the shortest tank line at ~105 ms
        kw['ediff'] = (kw['ediff'][0], float(feed[2:-6]))
    elif feed == 'allpass_noout':
        pass   # allpass feed into the tank, but not observed directly (feed-out fixed at 0)
    elif feed == 'raw':
        kw['ediff'] = None
    elif feed == 'none':
        kw = {}
    R = po.Reference(ref); x = np.zeros(NT); x[0] = 1.0
    lines, early, fed = model.render(x, st, basis=True, basis_feed=True, **kw)
    round6.ED = ed
    times = round6.extra_taps(n, xt); cols = round6.tap_basis(times, st) if n else []; k = len(cols)
    w0 = np.array(st['output_weights'], float).ravel() if seed == 0 else np.random.default_rng(100 + seed).normal(0, 1, 16)
    g0 = [float(ref[int((t + 2.5) * FS / 1000 + .5), c]) / .05 for c in range(2) for t in times[c]] if n else []
    w0 = np.r_[w0, 0.5 if not (feed == 'none' or feed.endswith('_noout')) else 0.0, 1.0, 0.0, g0]   # feed-out, early gain, crosstalk, extra taps
    six = slice(0, 6 * FS)
    C = np.stack(cols, axis=-1) if k else None

    def ir_of(w, sl=six):
        y = lines[sl] @ w[:16].reshape(2, 8).T + early[sl] * w[17] + fed[sl] * w[16] + early[sl][:, ::-1] * w[18]
        return y + (C[sl] @ w[19:] if k else 0)
    bounds = [(-8, 8)] * 16 + [(-4, 4) if not (feed == 'none' or feed.endswith('_noout')) else (0, 0), (-4, 4), (-1, 1) if xt else (0, 0)] + [(-4, 4)] * k
    t0 = time.time()
    res = minimize(lambda w: po.objective(ir_of(w), R)[0], w0, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=200))
    w = res.x; ir = ir_of(w, slice(0, NT))
    assert st.get('width', 1.0) == 1.0
    tag = f'd{dg}_ed{ed}_{feed}_n{n}_s{seed}' + (f'_x{xt}' if xt else '') + (f'_b{bw:g}' if bw else '') + (f'_w{bu:g}' if bu != 0.3 else '') + (f'_g{sg:g}' if sg else '') + ('_iwf' if iwf else '')
    np.save(WORK / 'round7' / f'{tag}-ir.npy', ir.astype(np.float32))
    gi = iter(w[19:])
    out = dict(candidate=tag, diffusion=dg, early_diffusion=ed, feed=feed, feed_out=float(w[16]), early_gain=float(w[17]), crosstalk=float(w[18]),
               extra_taps=[[[t, float(next(gi))] for t in times[c]] for c in range(2)] if n else None,
               output_weights=w[:16].reshape(2, 8).tolist(), fit_seconds=time.time() - t0, fit_message=str(res.message),
               objective=po.objective(ir[:6 * FS], R), ir=summary(ir[:6 * FS]), early_density=early_density.dips(ir),
               kurt=[early_density.kurt(ir, *w) for w in KW], kurt_ref=[early_density.kurt(ref, *w) for w in KW])
    data = round2.fresh40_data()
    ref12 = round2.r13.read(next(CAPTURES.glob('* - 6000ms.wav')))[round2.r13.FS:] / .25
    refs = [np.column_stack([fftconvolve(xx, ref12[:, c])[:len(xx)] for c in range(2)]) for _, xx in data]
    out['fresh40'] = round2.fresh40(resample_poly(ir, 1, 4, axis=0), data, refs)
    out['sources'] = round2.sources(ir)
    out['cases'] = round2.case_metrics(ir)
    (WORK / 'round7' / f'{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    print(tag, 'obj %.3f' % out['objective'][0], {kk: round(v, 3) for kk, v in out['objective'][1].items()},
          'fresh40 %.3f' % np.mean([r['envelope_db'] for r in out['fresh40']]),
          {kk: round(v, 3) for kk, v in out['sources'].items()},
          'early0-50 %.2f worst %.2f kurt %s (ref %s)' % (out['early_density']['mean_abs'], out['early_density']['worst'], [round(v, 1) for v in out['kurt']], [round(v, 1) for v in out['kurt_ref']]),
          flush=True)


if __name__ == '__main__':
    (WORK / 'round7').mkdir(parents=True, exist_ok=True)
    main()
