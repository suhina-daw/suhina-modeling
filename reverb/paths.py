# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Where the reverb research scripts read and write. Nothing is created on import.

  SUHINA_REPO             a checkout of suhina-daw/suhina-daw; the reverb model (model.py) and its
                          parity checks are kept there in scripts/reverb/model and are imported from it.
                          Required by every script that renders the model.
  SUHINA_REVERB_CAPTURES  folder of 48 kHz reference captures: '<name> - 6000ms.wav' (the reference
                          impulse response, the capture starts 1 s before the impulse) and
                          samples/ (dry stimuli and their captures, '<sound> 48k-stereo-20s - 6000ms.wav').
                          Captures are never committed; this has no default.
  SUHINA_REVERB_DATA      older research rounds the scripts can read (round 10, round 19 renders);
                          default: the parent of SUHINA_REVERB_CAPTURES' parent.
  SUHINA_REVERB_WORK      results, fitted IRs and listening packages; default target/reverb here.
  SUHINA_REVERB_PROBE     the native probe executable; default SUHINA_REPO's
                          target/release/examples/fdn_probe(.exe).

DATAFILES is data/ beside this file: small committed inputs (round 18's selected settings, the
E output weights) and the parameters of the candidates listened to (see README.md)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATAFILES = HERE / 'data'


def _env(name, default=None):
    v = os.environ.get(name)
    return Path(v) if v else default


class _Missing:
    """Stands in for an unset required path and fails on first use with the variable's name."""

    def __init__(self, name):
        self.name = name

    def _fail(self, *_):
        sys.exit(f'{self.name} is not set; see reverb/paths.py')

    __truediv__ = __fspath__ = __str__ = glob = exists = _fail


class _NoModel:
    """Turns a missing SUHINA_REPO into a clear message when a script imports the model."""

    def find_spec(self, name, path=None, target=None):
        if name == 'model':
            sys.exit('SUHINA_REPO is not set: the reverb model lives in suhina-daw/suhina-daw '
                     'scripts/reverb/model; see reverb/paths.py')
        return None


SUHINA_REPO = _env('SUHINA_REPO')
CAPTURES = _env('SUHINA_REVERB_CAPTURES') or _Missing('SUHINA_REVERB_CAPTURES')
DATA = _env('SUHINA_REVERB_DATA', CAPTURES.parent.parent if isinstance(CAPTURES, Path) else _Missing('SUHINA_REVERB_DATA'))
WORK = _env('SUHINA_REVERB_WORK', HERE.parent / 'target' / 'reverb')
_probe = 'fdn_probe.exe' if os.name == 'nt' else 'fdn_probe'
PROBE = _env('SUHINA_REVERB_PROBE', SUHINA_REPO / 'target' / 'release' / 'examples' / _probe if SUHINA_REPO else _Missing('SUHINA_REVERB_PROBE'))

# Appended, not prepended, so this folder's own modules win on a name clash.
if SUHINA_REPO:
    _model = str(SUHINA_REPO / 'scripts' / 'reverb' / 'model')
    if _model not in sys.path:
        sys.path.append(_model)
elif not any(isinstance(f, _NoModel) for f in sys.meta_path):
    sys.meta_path.append(_NoModel())
