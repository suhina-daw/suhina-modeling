# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Second REW-style page for one IR against the reference: spectrograms 0-1 s (both, mid channel),
L/R correlation over time (20 ms windows, the stereo image as it builds up), L-R phase difference
of the whole IR (1/6 oct, unwrapped mean), octave-band decay (EDC) 125 Hz-8 kHz, and C50/C80 per
octave. Writes compare2-<label>.png.
Usage: compare_ir2.py <label> <ir.npy>"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, spectrogram
O = Path(__file__).parent; FS = 48000
sys.path.insert(0, str(O))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
from metrics import ref
label, path = sys.argv[1], sys.argv[2]
h = np.load(path).astype(float)[:6 * FS]; h *= np.sqrt(np.sum(ref ** 2) / np.sum(h ** 2))
fig, ax = plt.subplots(4, 2, figsize=(18, 17))
for j, (name, x) in enumerate([('A ref', ref), (label, h)]):
    m = x[:FS].mean(axis=1); f, t, S = spectrogram(m, FS, nperseg=1024, noverlap=896)
    a = ax[0, j]; a.pcolormesh(t * 1000, f, 10 * np.log10(S + 1e-16), vmin=-140, vmax=-70, shading='auto')
    a.set_yscale('log'); a.set_ylim(60, 16000); a.set_title(f'Spectrogram 0-1 s, mid: {name}'); a.set_xlabel('ms')
w = 960
a = ax[1, 0]
for name, x in [('A ref', ref), (label, h)]:
    n = int(1.5 * FS) // w; f = x[:n * w].reshape(n, w, 2)
    cor = np.sum(f[:, :, 0] * f[:, :, 1], axis=1) / np.sqrt(np.sum(f[:, :, 0] ** 2, axis=1) * np.sum(f[:, :, 1] ** 2, axis=1) + 1e-30)
    a.plot(np.arange(n) * 20 + 10, cor, label=name)
a.set_title('L/R correlation in 20 ms windows (0 = wide, 1 = mono)'); a.set_xlabel('ms'); a.legend(); a.grid(alpha=.3)
a = ax[1, 1]
fr = np.fft.rfftfreq(len(ref), 1 / FS); fc = np.geomspace(50, 16000, 200)
for name, x in [('A ref', ref), (label, h)]:
    H = np.fft.rfft(x, axis=0); X = H[:, 0] * np.conj(H[:, 1])
    v = [np.angle(np.sum(X[(fr >= c * 2 ** (-1 / 12)) & (fr < c * 2 ** (1 / 12))])) for c in fc]
    co = [np.abs(np.sum(X[(fr >= c * 2 ** (-1 / 12)) & (fr < c * 2 ** (1 / 12))])) / np.sqrt(np.sum(np.abs(H[(fr >= c * 2 ** (-1 / 12)) & (fr < c * 2 ** (1 / 12)), 0]) ** 2) * np.sum(np.abs(H[(fr >= c * 2 ** (-1 / 12)) & (fr < c * 2 ** (1 / 12)), 1]) ** 2)) for c in fc]
    a.semilogx(fc, co, label=f'{name} coherence')
a.set_title('L/R coherence per 1/6 oct (whole IR); phase of a diffuse tail is random, coherence is what it means'); a.legend(); a.grid(alpha=.3, which='both')
OCT = [125, 250, 500, 1000, 2000, 4000, 8000]
for c in range(2):
    a = ax[2, c]
    for i, fo in enumerate(OCT):
        sos = butter(3, [fo / np.sqrt(2), min(fo * np.sqrt(2), 23000)], btype='bandpass', fs=FS, output='sos')
        for name, x, ls in [('A ref', ref, '-'), (label, h, '--')]:
            y = sosfilt(sos, x[:, c]); e = np.cumsum(y[::-1] ** 2)[::-1]; e = 10 * np.log10(e / e[0] + 1e-30)
            a.plot(np.arange(len(e))[::48] / FS, e[::48] - 6 * i, ls, color=f'C{i}', lw=1, label=f'{fo} Hz' if ls == '-' else None)
    a.set_xlim(0, 2); a.set_ylim(-60, 2); a.set_title(f'Octave EDC 0-2 s, {"LR"[c]} (solid ref, dashed {label}; offset 6 dB per band)'); a.legend(fontsize=7)
for c in range(2):
    a = ax[3, c]
    for name, x, mk in [('A ref', ref, 'o-'), (label, h, 's--')]:
        c50, c80 = [], []
        for fo in OCT:
            sos = butter(3, [fo / np.sqrt(2), min(fo * np.sqrt(2), 23000)], btype='bandpass', fs=FS, output='sos')
            y = sosfilt(sos, x[:, c]) ** 2
            c50.append(10 * np.log10(y[:int(.05 * FS)].sum() / y[int(.05 * FS):].sum()))
            c80.append(10 * np.log10(y[:int(.08 * FS)].sum() / y[int(.08 * FS):].sum()))
        a.semilogx(OCT, c50, mk, label=f'{name} C50'); a.semilogx(OCT, c80, mk, label=f'{name} C80', alpha=.6)
    a.set_title(f'Clarity per octave, {"LR"[c]} (time from the input impulse)'); a.legend(); a.grid(alpha=.3, which='both')
fig.tight_layout(); out = WORK / f'compare2-{label}.png'; fig.savefig(out, dpi=70); print(out)
