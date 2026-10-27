# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration checks of the production-device probe (set SUHINA_SATURATOR_PROBE)."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import audio


@unittest.skipUnless(os.environ.get('SUHINA_SATURATOR_PROBE'), 'native probe path not set')
class Probe(unittest.TestCase):
    def test_event_inside_block_preserves_prefix_and_changes_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t = np.arange(24000)/48000
            x = np.column_stack([.2*np.sin(2*np.pi*331*t), .1*np.sin(2*np.pi*557*t)])
            audio.write(root/'input.wav', 48000, x)
            outputs = []
            for name, changes in [('fixed', []), ('changed', [{'frame': 10003, 'param': 'type', 'value': 3}])]:
                config = root/f'{name}.json'
                config.write_text(json.dumps({'settings': {'drive': 12}, 'changes': changes}))
                output = root/f'{name}.wav'
                result = subprocess.run([os.environ['SUHINA_SATURATOR_PROBE'], str(root/'input.wav'),
                                         str(config), str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(audio.read(output)[1])
            a, b = outputs
            self.assertEqual(a.shape, x.shape)
            self.assertTrue(np.array_equal(a[:10003], b[:10003]))
            self.assertGreater(audio.rms(a[12000:]-b[12000:]), .01)
            self.assertGreater(audio.rms(a[:, 0]-a[:, 1]), .01)

    def test_invalid_schedule_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio.write(root/'input.wav', 48000, np.zeros((1000, 2)))
            cases = [({'unknown': 1}, []), ({'drive': 1000}, []),
                     ({}, [{'frame': 1001, 'param': 'type', 'value': 1}]),
                     ({}, [{'frame': 100, 'param': 'type', 'value': 1},
                           {'frame': 50, 'param': 'type', 'value': 2}])]
            for settings, changes in cases:
                config = root/'invalid.json'
                config.write_text(json.dumps({'settings': settings, 'changes': changes}))
                output = root/'output.wav'
                result = subprocess.run([os.environ['SUHINA_SATURATOR_PROBE'], str(root/'input.wav'),
                                         str(config), str(output)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
