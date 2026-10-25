# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment 4: how much do the selection metrics move under small random perturbations
of the incumbent's fitted output/input weights? Measures metric fragility, not quality."""
import copy, json, sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
from metrics import s
import run_five_rounds as r13

FS = 48000
x = np.zeros(10 * FS); x[0] = 1.0
data = r13.sources()
rng = np.random.default_rng(20261024)
rows = []
for kind in ['output', 'input']:
    for rel in [0.02, 0.05, 0.10]:
        for trial in range(3):
            t = copy.deepcopy(s)
            key = 'output_weights' if kind == 'output' else 'input_weights'
            w = np.array(t[key]); w = w * (1 + rel * rng.standard_normal(w.shape)); t[key] = w.tolist()
            ev = r13.evaluate(resample_poly(model.render(x, t), 1, 4, axis=0), data)
            rows.append(dict(kind=kind, rel=rel, trial=trial, score=ev['score'], orig=ev['original_envelope'],
                             pitch=ev['pitch_envelope'], vocal=ev['vocal_envelope']))
            print(rows[-1], flush=True)
(WORK / 'exp4-sensitivity.json').write_text(json.dumps(rows, indent=1))
