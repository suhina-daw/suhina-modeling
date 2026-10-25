# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""ISO 3382 data from a running REW (API on localhost:4735) for IRs already imported from
rew-irs/: generates RT60 per measurement in turn (REW runs one at a time) and tabulates
EDT, T30, C50, C80 and D50 per octave. An independent check of our own decay metrics."""
import json, time, urllib.request, urllib.error
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from paths import DATA, WORK, DATAFILES, CAPTURES, PROBE
B = 'http://localhost:4735'


def call(path, body=None):
    req = urllib.request.Request(B + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST' if body else 'GET')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode())


ms = call('/measurements'); out = {}
for idx in sorted(ms, key=int):
    title = ms[idx]['title']
    for _ in range(120):
        res = call(f'/measurements/{idx}/rt60?octaveFrac=1')
        if 'message' not in res:
            break
        msg = call(f'/measurements/{idx}/command', {'command': 'Generate RT60', 'parameters': {'octaveFrac': '1', 'filterOrder': '6'}})
        time.sleep(2)
    out[title] = res
(WORK / 'rew-rt60.json').write_text(json.dumps(out, indent=1))
keys = ['EDT', 'T30', 'C50', 'C80', 'D50']
for k in keys:
    print('==', k, '(octaves 125 250 500 1k 2k 4k)')
    for t, res in out.items():
        row = {r['fc']: r for r in res.values()} if isinstance(res, dict) and 'message' not in res else {}
        print(f'  {t:22s}', ' '.join(f"{row[f][k]:6.2f}" if f in row and k in row[f] else '   nan' for f in (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)))
