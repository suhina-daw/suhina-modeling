# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A level-matched listening package: five sources, the captured reference as A and each given
impulse response as B, C, D ... in folder order (the order the files sort in, so they can be
stacked in a DAW in the same order).

Each candidate is the dry source convolved with its IR, energy-matched once to A; then one common
max-300 ms-RMS target across the sources, with an estimated 4x-oversampled peak ceiling of -1 dBFS.
No limiter, AGC, onset shift or time-varying gain. Refuses to overwrite a package. Writes the
package to SUHINA_REVERB_WORK/<name>, a verification manifest beside it, and a '00 - START HERE.txt'
with the version descriptions and the optional notes.

Usage: make_listening.py <name> <label=ir.npy> [<label=ir.npy> ...] [--note TEXT]
  label  file-name label of the version (letters are assigned B, C, D ... in argument order)
  ir.npy a stereo 48 kHz IR (.npy), absolute or relative to SUHINA_REVERB_WORK
Example: make_listening.py listening-round8 "round7 D=round7/d0.3_ed0.0_raw_n0_s0_x1-ir.npy" \\
         "round8 s4=round8/iwfit_s4-ir.npy" "round8 s5=round8/iwfit_s5-ir.npy" --note "Basso (01)?"
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly
sys.path.insert(0, str(Path(__file__).parent))
from paths import WORK, CAPTURES

FS = 48000
SPECS = [('pitch', 'bass', 3, 'Bass upper'), ('pitch', 'sine', 0, 'Sine 262Hz'), ('original', 'kick', 0, 'Kick'),
         ('original', 'snare', 0, 'Snare'), ('original', 'vocals-male', 0, 'Vocals male')]


def load(p):
    sr, x = wavfile.read(p)
    if sr != FS or x.ndim != 2:
        sys.exit(f'{p}: expected a {FS} Hz stereo file')
    return x.astype(float)


def maxrms(x):
    e = np.mean(x * x, axis=1); c = np.r_[0, np.cumsum(e)]; w = int(.3 * FS)
    return np.sqrt(np.max((c[w:] - c[:-w]) / w))


def peak(x):
    return max(np.max(abs(x)), np.max(abs(resample_poly(x, 4, 1, axis=0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('name'); ap.add_argument('versions', nargs='+'); ap.add_argument('--note', action='append', default=[])
    a = ap.parse_args()
    dest = WORK / a.name
    if dest.exists():
        sys.exit(f'{dest} exists; not overwriting')
    variants = []
    for letter, v in zip('BCDEFGHIJ', a.versions):
        label, _, f = v.partition('=')
        p = Path(f) if Path(f).is_absolute() else WORK / f
        variants.append((letter, label, np.load(p).astype(float), p))
    groups, hashes = [], {}
    for group, sound, slot, label in SPECS:
        src = CAPTURES / 'samples' / ('pitch' if group == 'pitch' else '')
        dp = next(p for p in src.glob(sound + '*.wav') if 'dry' in p.name)
        cp = src / f'{sound} 48k-stereo-20s - 6000ms.wav'
        for p in (dp, cp):
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        dry = load(dp).mean(axis=1); ref = load(cp)
        n = int(17.5 * FS) if group == 'pitch' else len(ref); sl = slice(slot * n, (slot + 1) * n)
        items = [ref[sl]]
        for _, _, h, _ in variants:
            y = np.column_stack([fftconvolve(dry, h[:, c])[:len(ref)] for c in range(2)])[sl]
            y *= np.sqrt(np.sum(items[0] ** 2) / np.sum(y * y)); items.append(y)
        groups.append((label, items, max(maxrms(x) for x in items), max(peak(x) for x in items)))
        print('rendered', label, flush=True)
    target = min(rms * 10 ** (-1 / 20) / pk for _, _, rms, pk in groups)
    dest.mkdir(parents=True); manifest = []
    names = [('A', 'Captured reference')] + [(l, k) for l, k, _, _ in variants]
    for i, (label, items, rms, pk) in enumerate(groups, 1):
        gain = target / rms
        for (letter, kind), x in zip(names, items):
            path = dest / f'{i:02} - {label} 6000ms - {letter} {kind}.wav'
            y = (x * gain).astype(np.float32); wavfile.write(path, FS, y); p = peak(y)
            if p > 10 ** (-.99 / 20):
                sys.exit(f'{path.name}: peak {20 * np.log10(p):.2f} dBFS over the -1 dBFS ceiling')
            manifest.append({'file': path.name, 'group_gain_db': float(20 * np.log10(gain)), 'peak_4x_dbfs': float(20 * np.log10(p)),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (WORK / f'{a.name}-verification.json').write_text(json.dumps({
        'target_active_rms_db': float(20 * np.log10(target)), 'source_sha256': hashes,
        'versions': {l: str(p) for l, _, _, p in variants}, 'files': manifest}, indent=1))
    lines = [f'{a.name} - {len(SPECS)} naytetta x {len(names)} versiota', 'A = referenssi']
    lines += [f'{l} = {k}' for l, k, _, _ in variants] + [''] + a.note
    lines += ['Sama tasomenetelma kuin aiemmin. Mallirenderit.', '']
    (dest / '00 - START HERE.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('files', len(manifest), 'target', 20 * np.log10(target))


if __name__ == '__main__':
    main()
