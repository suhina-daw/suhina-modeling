# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Round 8: fit the tank's input weights together with its output weights. Output weights alone
cannot move when the tank's arrivals come (round 7: line 4's concentrated second-order arrival at
~105 ms, and the bass growing 100-200 ms instead of 200-400 ms). The tank is linear, so each line's
response to a unit injection is rendered once (diffused input and early feed separately, 2 x 8
renders) and the IR is bilinear in input and output weights:
    ir = sum_i iw_i (D_i @ wo) + |iw_i| (F_i @ wo) + early * g_e + swapped early * g_x
(the native feed scales each line's injection by |input weight|, as the model does).
Structure as round 7's d0.3_ed0_ap0.6_noout_x1: input diffusion .3, discrete early, allpass feed
(g .6) into the tank without direct output, L/R crosstalk; burst term 1, build-up window 0.6 s.
Usage: round8.py <seed> [burst weight, default 1]   (seed 0 starts from round 18's input weights and E's output weights)"""
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
import round2
import early_density

FS = 48000; NT = 10 * FS; SIX = 6 * FS
KW = [(.0025, .05), (.05, .15), (.3, 1)]
po.set_buildup(2.0, 0.6); po.set_burst(1.0)


def main():
    seed = int(sys.argv[1])
    bw = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    po.set_burst(bw)
    over, kw = round2.CANDIDATES['e']; kw = dict(kw); kw['ediff'] = (kw['ediff'][0], 0.6)
    st = dict(s, **over, early_diffusion=0.0, diffusion=0.3)
    x = np.zeros(NT); x[0] = 1.0
    D, F = [], []
    for i in range(8):
        si = dict(st, input_weights=[1.0 if j == i else 0.0 for j in range(8)])
        D.append(model.render(x, si, basis=True, **dict(kw, k=0.0))[0].astype(np.float32))
        Li, early = model.render(x, si, basis=True, mono_keep=0.0, **kw)
        F.append(Li.astype(np.float32))
        print('basis line', i, flush=True)
    D = np.stack(D, axis=1); F = np.stack(F, axis=1)   # (n, 8 inputs, 8 lines)
    R = po.Reference(ref)
    iw0 = np.array(st['input_weights'], float) if seed == 0 else np.random.default_rng(200 + seed).normal(0, 1, 8)
    wo0 = np.array(st['output_weights'], float).ravel() if seed == 0 else np.random.default_rng(100 + seed).normal(0, 1, 16)
    w0 = np.r_[wo0, iw0, 1.0, 0.0]

    def ir_of(w, sl=slice(0, SIX)):
        wo = w[:16].reshape(2, 8); iw = w[16:24]
        lines = np.tensordot(D[sl], iw, axes=([1], [0])) + np.tensordot(F[sl], np.abs(iw), axes=([1], [0]))
        return lines @ wo.T + early[sl] * w[24] + early[sl][:, ::-1] * w[25]
    bounds = [(-8, 8)] * 16 + [(-3, 3)] * 8 + [(-4, 4), (-1, 1)]
    t0 = time.time()
    res = minimize(lambda w: po.objective(ir_of(w), R)[0], w0, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=250))
    w = res.x; ir = ir_of(w, slice(0, NT)).astype(float)
    tag = f'iwfit_s{seed}' + (f'_b{bw:g}' if bw != 1.0 else '')
    np.save(WORK / 'round8' / f'{tag}-ir.npy', ir.astype(np.float32))
    out = dict(candidate=tag, structure='d0.3_ed0_ap0.6_noout_x1', output_weights=w[:16].reshape(2, 8).tolist(), input_weights=w[16:24].tolist(),
               early_gain=float(w[24]), crosstalk=float(w[25]), fit_seconds=time.time() - t0, fit_message=str(res.message),
               objective=po.objective(ir[:SIX], R), ir=summary(ir[:SIX]), early_density=early_density.dips(ir),
               kurt=[early_density.kurt(ir, *k) for k in KW], kurt_ref=[early_density.kurt(ref, *k) for k in KW])
    data = round2.fresh40_data()
    ref12 = round2.r13.read(next(CAPTURES.glob('* - 6000ms.wav')))[round2.r13.FS:] / .25
    refs = [np.column_stack([fftconvolve(xx, ref12[:, c])[:len(xx)] for c in range(2)]) for _, xx in data]
    out['fresh40'] = round2.fresh40(resample_poly(ir, 1, 4, axis=0), data, refs)
    out['sources'] = round2.sources(ir)
    out['cases'] = round2.case_metrics(ir)
    (WORK / 'round8' / f'{tag}.json').write_text(json.dumps(out, indent=1, default=float))
    print(tag, 'obj %.3f' % out['objective'][0], {k: round(v, 3) for k, v in out['objective'][1].items()},
          'fresh40 %.3f' % np.mean([r['envelope_db'] for r in out['fresh40']]), {k: round(v, 3) for k, v in out['sources'].items()},
          'kurt', [round(v, 1) for v in out['kurt']], 'iw', np.round(w[16:24], 2), flush=True)


if __name__ == '__main__':
    (WORK / 'round8').mkdir(parents=True, exist_ok=True)
    main()
