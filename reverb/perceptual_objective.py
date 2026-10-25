# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import-safe IR-level objective built from band-smoothed, attribute-level comparisons.

Every term compares the candidate with the reference's own value; nothing fits the
single-frequency fine structure. Inputs are stereo IRs at 48 kHz starting at the input
impulse (predelay included), energy-normalised to the reference before comparison.

Terms (each divided by a fixed scale so a typical value is near 1):
  edr      1/3-octave energy decay relief error in dB, 0-3 s, above -50 dB (Jot-style)
  early    1/3-octave band energy error over 0-80 ms and 80-300 ms (onset/body balance)
  color    |H| std/mean in 1/3-octave bands, 0.2-2 s, vs the reference (Rayleigh ~0.523)
  recur    decay-compensated autocorrelation peak above the reference's own peak
  stereo   1/3-octave interchannel correlation and L/R balance of the late tail
"""
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE

FS = 48000
CENTRES = 63 * 2 ** (np.arange(0, 22) / 3)          # 63 Hz .. ~8 kHz
EDGES = np.c_[CENTRES * 2 ** (-1 / 6), CENTRES * 2 ** (1 / 6)]
SCALES = dict(edr=1.0, early=1.0, color=0.05, recur=0.03, stereo=0.1, buildup=3.0, burst=1.0)
WEIGHTS = dict(edr=1.0, early=1.0, color=1.0, recur=1.0, stereo=0.5, buildup=0.0, burst=0.0)
# Build-up: octave-band 10 ms envelopes over 0-300 ms (mid and side), 30 ms smoothing.
# Weight 0 keeps earlier objective values reproducible; set_buildup(w) enables it.
OCT = [(88, 177), (177, 354), (354, 707), (707, 1414), (1414, 2828), (2828, 5657), (5657, 11314)]


BU_T = 0.3   # build-up window (s); the reference's bass grows until ~400 ms, so later rounds use 0.6


def set_buildup(weight, window=None):
    global BU_T
    WEIGHTS['buildup'] = weight
    if window is not None:
        BU_T = window


# Burst: broadband 5 ms mid and side envelopes over 10-300 ms; mean excess over the reference beyond
# 3 dB. The octave build-up term (30 ms smoothing) let a single concentrated tank arrival through
# (+7 dB at ~105 ms, one line's first arrival given a large output weight). Weight 0 by default.
def set_burst(weight):
    WEIGHTS['burst'] = weight


def _burst_env(x):
    s = x[int(.01 * FS):int(.3 * FS)]
    ms = np.stack([(s[:, 0] + s[:, 1]) / 2, (s[:, 0] - s[:, 1]) / 2], axis=1)
    n = len(ms) // 240
    return 10 * np.log10(np.mean(ms[:n * 240].reshape(n, 240, 2) ** 2, axis=1) + 1e-30)


def _band_masks(n):
    f = np.fft.rfftfreq(n, 1 / FS)
    return [(f >= lo) & (f < hi) for lo, hi in EDGES]


class Reference:
    """Precomputes everything about the reference once."""

    def __init__(self, ref, decay_seconds=5.321):
        self.ref = np.asarray(ref, float)
        self.energy = np.sum(self.ref ** 2)
        self.decay = decay_seconds
        self.n_edr = 1 << 11
        self.hop = 1 << 9
        self.feat = features(self.ref, self)


def _stft_bands(x, R):
    """Mono-summed power per 1/3-octave band per frame (frames of 2048, hop 512), 0-3 s."""
    m = x[:3 * FS].sum(axis=1)
    nfr = (len(m) - R.n_edr) // R.hop
    idx = np.arange(R.n_edr)[None, :] + R.hop * np.arange(nfr)[:, None]
    spec = np.abs(np.fft.rfft(m[idx] * np.hanning(R.n_edr), axis=1)) ** 2
    masks = _band_masks(R.n_edr)
    return np.stack([spec[:, mk].sum(axis=1) for mk in masks], axis=1) + 1e-30


def features(x, R):
    out = {}
    # EDR: backward-integrated band energy, dB normalised per band to its start.
    bands = _stft_bands(x, R)
    edr = np.cumsum(bands[::-1], axis=0)[::-1]
    out['edr_db'] = 10 * np.log10(edr / edr[:1])
    out['band_level_db'] = 10 * np.log10(edr[0])
    # Early balance per band (stereo-summed energy).
    def seg(t0, t1):
        s = x[int(t0 * FS):int(t1 * FS)]
        p = np.sum(np.abs(np.fft.rfft(s, n=1 << 15, axis=0)) ** 2, axis=1)
        return 10 * np.log10(np.array([p[mk].sum() for mk in _band_masks(1 << 15)]) + 1e-30)
    out['early'] = np.r_[seg(0, .08), seg(.08, .3)]
    # Colour: magnitude spread within each band of the late tail, per channel.
    late = x[int(.2 * FS):2 * FS]
    H = np.abs(np.fft.rfft(late, n=1 << 18, axis=0))
    masks = _band_masks(1 << 18)
    out['color'] = np.array([[H[mk, c].std() / max(H[mk, c].mean(), 1e-30) for mk in masks] for c in range(2)])
    # Recurrence: decay-compensated autocorrelation, peak beyond 5 ms.
    lo, hi = int(.1 * FS), 3 * FS
    z = x[lo:hi] * np.exp(np.arange(hi - lo)[:, None] / FS * 3 * np.log(10) / R.decay)
    n = 1 << 19
    ac = np.fft.irfft(np.sum(np.abs(np.fft.rfft(z, n=n, axis=0)) ** 2, axis=1), n=n)[:FS // 2]
    ac /= max(ac[0], 1e-30)
    out['recur'] = float(np.max(np.abs(ac[int(.005 * FS):])))
    # Stereo: band correlation and balance of the late tail.
    Hc = np.fft.rfft(late, n=1 << 18, axis=0)
    cor, bal = [], []
    for mk in masks:
        a, b = Hc[mk, 0], Hc[mk, 1]
        ea, eb = np.sum(np.abs(a) ** 2), np.sum(np.abs(b) ** 2)
        cor.append(np.real(np.sum(a * np.conj(b))) / max(np.sqrt(ea * eb), 1e-30))
        bal.append(10 * np.log10(max(ea, 1e-30) / max(eb, 1e-30)))
    out['stereo'] = np.r_[cor, np.array(bal) / 10]
    if WEIGHTS['buildup']:
        from scipy.signal import butter, sosfilt
        seg = x[:int((BU_T + .03) * FS)]
        ms = np.stack([(seg[:, 0] + seg[:, 1]) / 2, (seg[:, 0] - seg[:, 1]) / 2], axis=1)
        env = []
        for lo, hi in OCT:
            z = sosfilt(butter(2, [lo, hi], btype='bandpass', fs=FS, output='sos'), ms, axis=0)[:int(BU_T * FS)]
            e = np.mean(z.reshape(-1, 480, 2) ** 2, axis=1)
            e = np.convolve(np.pad(e[:, 0], 1, mode='edge'), np.ones(3) / 3, 'valid'), np.convolve(np.pad(e[:, 1], 1, mode='edge'), np.ones(3) / 3, 'valid')
            env.append(10 * np.log10(np.stack(e, axis=1) + 1e-30))
        out['buildup'] = np.stack(env)
    return out


def terms(x, R):
    x = np.asarray(x, float)
    x = x * np.sqrt(R.energy / max(np.sum(x ** 2), 1e-30))
    f, r = features(x, R), R.feat
    m = r['edr_db'] > -50
    t = dict(
        edr=float(np.mean(np.abs(f['edr_db'][m] - r['edr_db'][m]))
                  + np.mean(np.abs((f['band_level_db'] - f['band_level_db'].mean())
                                   - (r['band_level_db'] - r['band_level_db'].mean())))),
        early=float(np.mean(np.abs(f['early'] - r['early']))),
        color=float(np.mean(np.abs(f['color'] - r['color']))),
        recur=float(max(0.0, f['recur'] - r['recur'])),
        stereo=float(np.mean(np.abs(f['stereo'] - r['stereo']))),
    )
    if WEIGHTS['buildup']:
        if 'buildup' not in r:
            r.update(features(R.ref * 1.0, R) if False else {'buildup': features(R.ref, R)['buildup']})
        rb = r['buildup']; mk = rb > rb.max(axis=1, keepdims=True) - 40
        t['buildup'] = float(np.mean(np.abs(f['buildup'] - rb)[mk]))
    if WEIGHTS['burst']:
        if 'burst' not in r:
            r['burst'] = _burst_env(R.ref * 1.0)
        t['burst'] = float(np.mean(np.maximum(0.0, _burst_env(x) - r['burst'] - 3.0)))
    return t


def objective(x, R):
    t = terms(x, R)
    return sum(WEIGHTS[k] * t[k] / SCALES[k] for k in t), t
