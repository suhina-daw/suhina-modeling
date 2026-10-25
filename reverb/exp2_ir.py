# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment 2: early pattern -> per-channel allpass diffusion -> tank."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
from metrics import *

CHAINS = {
    'short': ([113, 211, 337], [127, 229, 359]),
    'mid': ([241, 379, 563, 797], [263, 409, 601, 853]),
    'long': ([443, 691, 1009, 1361], [479, 727, 1061, 1427]),
}
results = {}
for cname, delays in CHAINS.items():
    for gd in [0.5, 0.7]:
        for r in [1, 4]:
            for keep in [1, 0]:
                k = float(np.sqrt(r / E_EARLY))
                name = f'{cname}_g{gd}_r{r}_keep{keep}'
                ir = model.render(x, s, variant='early_lr', k=k, mono_keep=keep, ediff=(delays, gd))
                results[name] = summary(ir)
                m = results[name]
                print(name, round(m['density_mae_50_700ms'], 3), round(m['density_mean_200_500ms'], 3),
                      round(m['corr_mean_50_500ms'], 3), round(m['recurrence_max_abs'], 3), m['segment_energy_vs_ref_db']['0-0.05'])
(WORK / 'exp2-results.json').write_text(json.dumps(results, indent=1))
