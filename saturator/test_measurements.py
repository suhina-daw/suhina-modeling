# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic tests: no reference captures required."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from . import audio
from .project import assemble, key, type_changes, automation_events
from .waveshaper_captures import capture


class Measurements(unittest.TestCase):
    def test_integer_and_fractional_delay(self):
        rng = np.random.default_rng(431)
        noise = signal.sosfilt(signal.butter(5, .12, output='sos'), rng.normal(size=8000))
        noise[:500] = noise[-500:] = 0
        x = np.column_stack([noise, noise])
        for lag in [9, -7, 3.35, -2.7]:
            y = audio.advance(x, -lag)
            found = audio.delay(x, y)
            self.assertAlmostEqual(found, lag, delta=.025)
            self.assertLess(audio.rms(audio.advance(y, found)[1000:-1000]-x[1000:-1000]), .002)

    def test_gain_and_silence_do_not_dilute_measurements(self):
        x = np.zeros((48000, 2)); x[10000:20000] = .1
        mask = audio.active_mask(x)
        result = audio.metrics(x, x*2, mask)
        self.assertAlmostEqual(result['gain_error_db'], 6.020599913, places=6)
        self.assertLess(mask.sum(), 11000)
        self.assertEqual(audio.metrics(x, x, np.zeros(len(x), bool))['status'], 'silent-input')

    def test_known_cubic_harmonics(self):
        t = np.arange(24000)/48000
        x = np.sin(2*np.pi*400*t)
        clean = audio.harmonics(x*.2, 48000, 400)
        self.assertLess(clean['thd'], 1e-10)
        cubic = audio.harmonics(x + .2*x**3, 48000, 400)
        self.assertAlmostEqual(cubic['amplitudes'][0], 1.15, places=9)
        self.assertAlmostEqual(cubic['amplitudes'][2], .05, places=9)

    def test_io_and_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.wav'
            x = np.ones((480, 2), dtype=np.int16)*8192
            wavfile.write(path, 48000, x)
            _, y = audio.read(path)
            self.assertTrue(np.all(y == .25))
            with self.assertRaises(FileExistsError):
                audio.write(path, 48000, y)
            wavfile.write(path, 44100, x)
            with self.assertRaises(ValueError): audio.read(path)
            wavfile.write(path, 48000, x[:, 0])
            with self.assertRaises(ValueError): audio.read(path)

    def test_suppressed_fundamental_is_not_reported_as_thd(self):
        t = np.arange(24000)/48000
        doubled = audio.harmonics(np.sin(2*np.pi*800*t), 48000, 400)
        self.assertIsNone(doubled['thd'])
        self.assertEqual(doubled['thd_status'], 'fundamental-below-floor')
        self.assertAlmostEqual(doubled['amplitudes'][1], 1.0, places=9)
        for frequency in [0, -10, 24000, float('nan')]:
            with self.assertRaises(ValueError): audio.harmonics(t, 48000, frequency)

    def test_project_source_offsets_are_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.wav'
            audio.write(path, 48000, np.column_stack([np.arange(1000)/1000]*2))
            clip = {'source': str(path), 'sha256': audio.digest(path),
                    'start': 100, 'end': 300, 'offset': 500, 'gain': .5}
            x = assemble({'frames': 500, 'clips': [clip]})
            self.assertAlmostEqual(x[100, 0], .25)
            self.assertTrue(np.all(x[:100] == 0))
            with self.assertRaises(ValueError): assemble({'frames': 500, 'clips': [clip, clip]})
            clip['sha256'] = 'wrong'
            with self.assertRaises(ValueError): assemble({'frames': 500, 'clips': [clip]})

    def test_long_and_short_capture_names_do_not_collide(self):
        self.assertEqual(key('3 sine steps-48k-stereo-drive12db'), 'sine steps')
        self.assertEqual(key('16 sine-dry 48k-stereo-drive12db'), 'sine')

    def test_export_track_number_is_not_a_number_in_the_project_title(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sine = folder/'Waveshaper C - Curve 5 Linear 100 3 sine steps-48k.wav'
            bass = folder/'Waveshaper C - Curve 5 Linear 100 5 bass steps-48k.wav'
            sine.touch(); bass.touch()
            self.assertEqual(capture(folder, '03'), sine)
            self.assertEqual(capture(folder, '05'), bass)
            with self.assertRaises(ValueError): capture(folder, '02')
            (folder/'- 3 sine steps-48k.wav').touch()
            with self.assertRaises(ValueError): capture(folder, '03')

    def test_both_mode_durations_and_exact_automation_events(self):
        for seconds in [2, 16]:
            events = [(-1000, 0)] + [(i*seconds*2, i) for i in range(8)]
            changes = type_changes(events, 120)
            self.assertEqual([e['frame'] for e in changes], [i*seconds*48000 for i in range(8)])
        events = [(i*32, i) for i in range(8)]
        events[1] = (31.715695, 1)
        self.assertEqual(type_changes(events, 120)[1]['frame'], round(31.715695*24000))
        with self.assertRaises(ValueError): type_changes([(0, 0), (4, 1.5)], 120)
        with self.assertRaises(ValueError): type_changes([(0, 0), (4, 9)], 120)

    def test_other_parameter_automation_is_never_silently_ignored(self):
        track = ET.fromstring('<Track><AutomationEnvelope><EnvelopeTarget><PointeeId Value="12"/>'
                              '</EnvelopeTarget><Automation><Events><EnumEvent Time="0" Value="0"/>'
                              '</Events></Automation></AutomationEnvelope></Track>')
        self.assertEqual(automation_events(track, '12'), [(0.0, 0.0)])
        with self.assertRaises(ValueError): automation_events(track, '13')


if __name__ == '__main__':
    unittest.main()
