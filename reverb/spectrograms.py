# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""0-2 s spectrograms of the IRs in rew-irs (A-D), mid and side, same scale; plus the
0-300 ms build-up per octave. What REW's spectrogram view shows, drawn here."""
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import stft, butter, sosfilt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
O = Path(__file__).parent
names = ['A ref', 'B R18', 'C E', 'D uusi (round4 C)']
irs = {n: wavfile.read(WORK / 'rew-irs' / f'{n}.wav')[1].astype(float) for n in names}
fig, axes = plt.subplots(4, 2, figsize=(15, 14), layout='constrained', sharex=True, sharey=True)
for row, n in enumerate(names):
    x = irs[n][:2 * 48000]
    for col, sig in enumerate([(x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2]):
        f, t, Z = stft(sig, 48000, nperseg=2048, noverlap=2048 - 128)
        S = 20 * np.log10(np.abs(Z) + 1e-9)
        im = axes[row, col].pcolormesh(t, f, S, vmin=-110, vmax=-40, shading='auto', cmap='magma')
        axes[row, col].set(yscale='symlog', ylim=(60, 16000), title=f'{n} - {"mid" if col == 0 else "side"}')
for ax in axes[-1]: ax.set_xlabel('s')
fig.colorbar(im, ax=axes, shrink=.5, label='dB')
fig.savefig(WORK / 'spectrograms-0-2s.png', dpi=80)
fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained', sharey=True)
for ax, (lo, hi) in zip(axes.flat, [(88, 177), (177, 354), (354, 707), (707, 1414), (1414, 2828), (2828, 5657), (5657, 11314)]):
    for n in names:
        m = irs[n][:int(.3 * 48000)].sum(axis=1)
        z = sosfilt(butter(2, [lo, hi], btype='bandpass', fs=48000, output='sos'), m)
        e = 10 * np.log10(np.mean(z.reshape(-1, 240) ** 2, axis=1) + 1e-20)
        ax.plot(np.arange(len(e)) * 5, e - np.max(e), label=n, lw=1)
    ax.set(title=f'{lo}-{hi} Hz', xlabel='ms', ylim=(-40, 3)); ax.grid(alpha=.3)
axes.flat[0].legend(fontsize=8); axes.flat[-1].axis('off')
fig.savefig(WORK / 'buildup-0-300ms.png', dpi=80)
