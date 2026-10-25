# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Writes the native settings for candidate E and checks native against the model."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
O = Path(__file__).parent
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
from metrics import s, E_EARLY
from fit_refit import CANDIDATES, LONG

EXE = PROBE
W = json.loads((DATAFILES / 'e-output-weights.json').read_text())['output_weights']
e = dict(s, output_weights=W, early_feed=dict(gain=float(np.sqrt(1 / E_EARLY)), allpass_gain=0.5,
         allpass_ms=[[d / 48 for d in ch] for ch in LONG]))
path = WORK / 'candidate-e-settings.json'
path.write_text(json.dumps({'long': e, 'benchmark': e}, indent=1))
dest = WORK / 'native-e'
subprocess.run([str(EXE), str(dest), '--flat-decay', '--diffused', '--render-only', '--impulse-only', '--settings', str(path)], check=True)
sr, y = wavfile.read(dest / 'long-impulse-wet.wav'); y = y.astype(float)
x = np.zeros(len(y)); x[48000] = .25
m = model.render(x, dict(s, output_weights=W), **CANDIDATES['firdiff_long_g0.5_lr_r1'])
print('native vs model max rel err', np.max(np.abs(m - y)) / np.max(np.abs(y)), 'rel rms', np.sqrt(np.mean((m - y) ** 2) / np.mean(y ** 2)))
