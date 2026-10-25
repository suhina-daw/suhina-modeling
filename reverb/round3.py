# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Round 3: the kick showed a 15-20 dB hole at 70-120 ms between the early pattern and the
first late arrivals. Fits with the build-up term enabled, without and with the fed early
pattern observed directly (two extra per-channel gains). Same four evaluations as round 2.
Usage: round3.py <e|e_fedout> <seed>   (seed 0 starts from E's weights)"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample_poly
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
import perceptual_objective as po
from metrics import s, ref, summary
import round2

po.set_buildup(1.0)
FS = 48000


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    over, kw = round2.CANDIDATES['e']; st = dict(s, **over)
    fedout = name == 'e_fedout'
    R = po.Reference(ref); x = np.zeros(6 * FS); x[0] = 1.0
    lines, early, fed = model.render(x, st, basis=True, basis_feed=True, **kw)
    w0 = np.array(st['output_weights'], float).ravel() if seed == 0 else np.random.default_rng(100 + seed).normal(0, 1, 16)
    if fedout:
        w0 = np.r_[w0, 0.0, 0.0]

    def ir_of(w):
        y = lines @ w[:16].reshape(2, 8).T + early
        return y + fed * w[16:18][None, :] if fedout else y
    t0 = time.time()
    res = minimize(lambda w: po.objective(ir_of(w), R)[0], w0, method='L-BFGS-B',
                   bounds=[(-8, 8)] * len(w0), options=dict(maxiter=200))
    st['output_weights'] = res.x[:16].reshape(2, 8).tolist()
    fo = res.x[16:18].tolist() if fedout else None
    ir = model.render(np.r_[1.0, np.zeros(10 * FS - 1)], st, feed_out=fo, **kw)
    tag = f'{name}_s{seed}'
    np.save(WORK / 'round3' / f'{tag}-ir.npy', ir.astype(np.float32))
    out = dict(candidate=tag, feed_out=fo, output_weights=st['output_weights'], fit_seconds=time.time() - t0,
               objective=po.objective(ir[:6 * FS], R), ir=summary(ir[:6 * FS]))
    data = round2.fresh40_data()
    ref12 = round2.r13.read(next(CAPTURES.glob('* - 6000ms.wav')))[round2.r13.FS:] / .25
    from scipy.signal import fftconvolve
    refs = [np.column_stack([fftconvolve(xx, ref12[:, c])[:len(xx)] for c in range(2)]) for _, xx in data]
    out['fresh40'] = round2.fresh40(resample_poly(ir, 1, 4, axis=0), data, refs)
    out['sources'] = round2.sources(ir)
    out['cases'] = round2.case_metrics(ir)
    (WORK / 'round3' / f'{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    print(tag, 'obj %.3f' % out['objective'][0], {k: round(v, 3) for k, v in out['objective'][1].items()},
          'fresh40 %.3f' % np.mean([r['envelope_db'] for r in out['fresh40']]),
          {k: round(v, 3) for k, v in out['sources'].items()}, 'fo', None if fo is None else np.round(fo, 3), flush=True)


if __name__ == '__main__':
    (WORK / 'round3').mkdir(parents=True, exist_ok=True)
    main()
