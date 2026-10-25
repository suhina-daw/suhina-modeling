# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ranks round 2 multi-start fits by objective + sine-tail roughness match + L/R balance."""
import json, glob, os
import numpy as np
from roughness import roughness
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
ref = np.load('round2/roughness.npz')['ref']
def rstats(r): return np.r_[r.mean(0), np.percentile(r, 90, axis=0)]
rs = rstats(ref); rows = []
for f in sorted(glob.glob(str(WORK / 'round2' / '*.json'))):
    n = os.path.basename(f)[:-5]
    if not (n == 'e' or n == 'r18' or '_seed' in n): continue
    d = json.load(open(f)); r = roughness(np.load(f'round2/{n}-ir.npy').astype(float))
    rows.append(dict(name=n, obj=d['objective'][0], rough_err=float(np.mean(np.abs(rstats(r) - rs))),
                     imbalance=float(abs(r[:, 0].mean() - r[:, 1].mean())),
                     fresh40=float(np.mean([c['envelope_db'] for c in d['fresh40']])), **d['sources'],
                     density=d['ir']['density_mae_50_700ms'], recurrence=d['ir']['recurrence_max_abs']))
for r in rows: r['rank'] = r['obj'] + 2 * r['rough_err'] + 2 * r['imbalance']
rows.sort(key=lambda r: r['rank'])
print('ref roughness stats', rs.round(2))
for r in rows:
    print(f"{r['name']:12s} rank {r['rank']:.2f} obj {r['obj']:.2f} roughErr {r['rough_err']:.2f} imbal {r['imbalance']:.2f} fresh40 {r['fresh40']:.3f} orig {r['orig_env']:.3f} pitch {r['pitch_env']:.3f} vocal {r['vocal_env']:.3f} pmod {r['pitch_mod']:.2f} dens {r['density']:.3f} rec {r['recurrence']:.3f}")
json.dump(rows, open('round2/selection.json', 'w'), indent=1)
