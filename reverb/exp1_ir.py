# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment 1: IR-level effect of feeding the stereo early pattern into the tank."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
from ir_diagnostics import ir_diagnostics

from metrics import *

results = {'reference': summary(ref)}
results['reference']['density_mae_50_700ms'] = 0.0
variants = [('base', 'base', 0, 1)]
for r in [0.25, 1, 4]:
    k = float(np.sqrt(r / E_EARLY))
    variants += [(f'early_mono_r{r}', 'early_mono', k, 1), (f'early_lr_r{r}', 'early_lr', k, 1)]
for r in [1, 4]:
    k = float(np.sqrt(r / E_EARLY))
    variants += [(f'early_lr_only_r{r}', 'early_lr', k, 0)]
irs = {}
for name, v, k, keep in variants:
    ir = model.render(x, s, variant=v, k=k, mono_keep=keep)
    irs[name] = ir
    results[name] = summary(ir)
    print(name, json.dumps({kk: vv for kk, vv in results[name].items() if kk != 'band_t60'}))
print('ref', results['reference'])
(WORK / 'exp1-results.json').write_text(json.dumps(results, indent=1))
np.savez_compressed(WORK / 'exp1-irs.npz', ref=ref, **irs)
