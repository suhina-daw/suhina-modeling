# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared IR metrics against the 6000 ms reference (import-safe)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
import model
from ir_diagnostics import ir_diagnostics

O = Path(__file__).parent
R = DATA
FS = 48000
s = json.loads((DATAFILES / 'selected-settings.json').read_text())['long']
sr, a = wavfile.read(next(CAPTURES.glob('* - 6000ms.wav')))
ref = a[FS:FS + 6 * FS].astype(float) / .25
N = len(ref)
x = np.zeros(N); x[0] = 1.0
E_EARLY = float(np.mean(np.sum(np.array(s['early_tap_weights']) ** 2, axis=1)))


def band_t60(ir):
    out = []
    for lo, hi in [(40, 120), (120, 350), (350, 1000), (1000, 3000), (3000, 7000), (7000, 16000)]:
        y = sosfilt(butter(3, [lo, hi], btype='bandpass', fs=FS, output='sos'), ir, axis=0)
        e = np.sum(y * y, axis=1); edc = np.cumsum(e[::-1])[::-1]; edc = 10 * np.log10(np.maximum(edc / edc[0], 1e-15))
        t = np.arange(len(edc)) / FS; m = (edc < -5) & (edc > -35); slope = np.polyfit(t[m], edc[m], 1)[0]
        out.append(-60 / slope)
    return out


def summary(ir):
    ir = ir * np.sqrt(np.sum(ref ** 2) / np.sum(ir ** 2))
    d = ir_diagnostics(ir)
    dens = np.array(d['threshold_density']); rd = np.array(ir_diagnostics(ref)['threshold_density'])
    win = slice(5, 70)  # 50-700 ms
    # early energy split: 0-50, 50-200, 200-700 ms, dB relative to ref
    def seg(y, t0, t1):
        return 10 * np.log10(np.sum(y[int(t0 * FS):int(t1 * FS)] ** 2, axis=0))
    segs = {f'{t0}-{t1}': (seg(ir, t0, t1) - seg(ref, t0, t1)).round(2).tolist() for t0, t1 in [(0, .05), (.05, .2), (.2, .7)]}
    return {
        'density_mae_50_700ms': float(np.mean(np.abs(dens[win] - rd[win]))),
        'density_mean_200_500ms': float(np.mean(dens[20:50])),
        'corr_mean_50_500ms': float(np.mean(np.array(d['correlation'])[5:50])),
        'recurrence_max_abs': float(max(abs(p['correlation']) for p in d['recurrence_peaks'])),
        'band_t60': [round(v, 3) for v in band_t60(ir)],
        'segment_energy_vs_ref_db': segs,
    }


