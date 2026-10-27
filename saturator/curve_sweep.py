# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check Curve 5/25/50 percent predictions on independent sine and triangle exports."""
import os
from pathlib import Path

import numpy as np
from scipy import signal

from . import audio
from .__main__ import fundamental, load, required, save
from .waveshaper_captures import capture


def run():
    root = required('SUHINA_SATURATOR_CAPTURES')
    work = Path(os.environ.get('SUHINA_SATURATOR_WORK', 'target/saturator'))
    inventory = load(work/'inventory.json')
    rows = []
    for ident in ['02', '03']:
        track = next(t for t in inventory['tracks'] if t['id'] == ident)
        _, dry = audio.read(work/'baseline'/f'{ident}-input.wav')
        dry = dry[:16*48000]
        driven = dry*10**(track['settings']['drive']/20)
        for letter, curve in [('B', .05), ('D', .25), ('E', .5)]:
            path = capture(root/f'waveshaper-{letter}', ident)
            sr, ref = audio.read(path)
            if len(ref) < len(dry):
                raise ValueError('at least the first 16 seconds are required')
            prediction = np.sin(2*np.pi*curve*driven**2)
            # Research-only approximation at the base sample rate. The production
            # comparison must use the Rust probe's actual oversampled path.
            pole = .999**(44100/sr)
            filtered = signal.lfilter([1, -1], [1, -pole], prediction, axis=0)
            filtered = signal.lfilter([1, -1], [1, -pole], filtered, axis=0)
            row = {'track': ident, 'curve': curve, 'path': str(path), 'sha256': audio.digest(path),
                   'sample_rate': sr, 'frames': len(ref), 'clips': []}
            for clip in track['clips'][:8]:
                lo = clip['start']+9600
                hi = min(clip['end']-2400, lo+24000)
                measurement = audio.metrics(ref[lo:hi], filtered[lo:hi], audio.active_mask(dry)[lo:hi])
                measurement.update({'start_frame': lo, 'end_frame': hi,
                                    'input_rms_dbfs': audio.db(audio.rms(dry[lo:hi]))})
                if ident == '03':
                    freq = fundamental(dry[lo:hi], sr)
                    if freq is None:
                        raise ValueError('sine source is not stationary')
                    observed = audio.harmonics(ref[lo:hi], sr, freq)
                    predicted = audio.harmonics(prediction[lo:hi], sr, freq)
                    a, b = np.asarray(observed['amplitudes']), np.asarray(predicted['amplitudes'])
                    measurement.update({'observed': observed, 'predicted': predicted,
                                        'amplitude_error_relative_db': audio.db(np.linalg.norm(a-b)/np.linalg.norm(a))})
                row['clips'].append(measurement)
            rows.append(row)
            print(ident, letter, 'checked', flush=True)
    save(work/'curve-sweep.json', {'rows': rows, 'limits': [
        'Predictive formula was fixed from the earlier 5-percent capture; no fitting to D/E.',
        'Research waveform has no oversampling and no delay correction.',
        'No conclusion about aliasing or Curve above 50 percent.']})


if __name__ == '__main__':
    run()
