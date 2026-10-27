# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure independent A/B/C exports; never fit using the musical holdouts.

Run with python -m saturator.waveshaper_captures. The capture and work roots
use the same environment variables as the main measurement commands.
"""
import os
import re
from pathlib import Path

import numpy as np

from . import audio
from .__main__ import fundamental, load, required, save


def capture(folder, track_id):
    matches = []
    for path in folder.glob('*.wav'):
        # Export names can include numbered settings before the track number.
        # The final number followed by a non-numeric name identifies the track.
        numbers = re.findall(r'(?:^| )(\d+) (?=\D)', path.stem)
        if numbers and int(numbers[-1]) == int(track_id):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f'{folder}: expected one export for track {track_id}, got {len(matches)}')
    return matches[0]


def run():
    captures = required('SUHINA_SATURATOR_CAPTURES')
    work = Path(os.environ.get('SUHINA_SATURATOR_WORK', 'target/saturator'))
    inventory = load(work/'inventory.json')
    rows = []
    for track in inventory['tracks']:
        paths = [capture(captures/f'waveshaper-{letter}', track['id']) for letter in 'ABC']
        arrays = [audio.read(p)[1] for p in paths]
        _, dry = audio.read(work/'baseline'/f"{track['id']}-input.wav")
        if any(x.shape != dry.shape for x in arrays):
            raise ValueError(f"track {track['id']}: inconsistent duration")
        # All A/B/C use the same arrangement. No relative shift or gain fitting:
        # C = 2*A + B tests additivity of the independently recorded controls.
        mask = audio.active_mask(dry)
        a, b, c = arrays
        row = {'id': track['id'], 'family': track['family'], 'holdout': track['holdout'],
               'captures': [{'path': str(p), 'sha256': audio.digest(p)} for p in paths],
               'c_minus_2a_minus_b_relative_db': audio.db(audio.rms((c-2*a-b)[mask]))-audio.db(audio.rms(c[mask])),
               'peak_b': float(np.max(np.abs(b))), 'sines': []}
        if track['family'] == 'sine steps':
            for clip in track['clips'][:8]:
                lo, hi = clip['start']+9600, min(clip['end']-2400, clip['start']+33600)
                freq = fundamental(dry[lo:hi], 48000)
                if freq is None:
                    continue
                row['sines'].append({'start_frame': lo, 'end_frame': hi,
                                     'input': audio.harmonics(dry[lo:hi], 48000, freq),
                                     'research_sine_of_squared_input': audio.harmonics(
                                         np.sin(2*np.pi*.05*(dry[lo:hi]*10**(track['settings']['drive']/20))**2),
                                         48000, freq),
                                     **{letter: audio.harmonics(x[lo:hi], 48000, freq)
                                        for letter, x in zip('ABC', arrays)}})
        rows.append(row)
        print(track['id'], f"C-(2A+B): {row['c_minus_2a_minus_b_relative_db']:.2f} dB", flush=True)
    save(work/'waveshaper-abc.json', {'tracks': rows,
         'limits': ['Settings must be checked in the saved A/B/C projects.',
                    'Additivity alone does not identify the nonlinear curve or filter.',
                    'The research prediction has no oversampling or DC filtering; compare harmonic amplitudes only.',
                    'Curve values beyond 0 and 0.05 are not validated.']})


if __name__ == '__main__':
    run()
