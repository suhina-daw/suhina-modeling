# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare every measured clip against a baseline, without averaging away regressions."""
import argparse
import os
from pathlib import Path

import numpy as np

from .__main__ import load, save


def compare_reports(baseline, candidate):
    rows = []
    if len(baseline['tracks']) != len(candidate['tracks']):
        raise ValueError('track counts differ')
    for old, new in zip(baseline['tracks'], candidate['tracks']):
        if (old['id'], old['family']) != (new['id'], new['family']):
            raise ValueError('track identities differ')
        if len(old['modes']) != len(new['modes']):
            raise ValueError('mode counts differ')
        for before, after in zip(old['modes'], new['modes']):
            if before['mode'] != after['mode'] or len(before['clips']) != len(after['clips']):
                raise ValueError('mode windows differ')
            pairs = []
            for a, b in zip(before['clips'], after['clips']):
                if any(a[k] != b[k] for k in ['clip', 'start_frame', 'end_frame', 'status']):
                    raise ValueError('clip windows differ')
                if a['status'] != 'measured':
                    continue
                pairs.append({'clip': a['clip'],
                              'baseline_error_db': a['error_relative_db'],
                              'candidate_error_db': b['error_relative_db'],
                              'error_change_db': b['error_relative_db']-a['error_relative_db'],
                              'absolute_gain_error_change_db': abs(b['gain_error_db'])-abs(a['gain_error_db']),
                              'spectral_error_change_db': (b['spectral_error_db']-a['spectral_error_db'])
                                  if a['spectral_error_db'] is not None and b['spectral_error_db'] is not None else None})
            rows.append({'id': old['id'], 'family': old['family'], 'holdout': old['holdout'],
                         'mode': before['mode'], 'clips': pairs})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    args = parser.parse_args()
    for name in [args.baseline, args.candidate]:
        if not name.replace('-', '').replace('_', '').isalnum():
            parser.error('use simple version labels')
    work = Path(os.environ.get('SUHINA_SATURATOR_WORK', 'target/saturator'))
    rows = compare_reports(load(work/args.baseline/'comparison.json'), load(work/args.candidate/'comparison.json'))
    save(work/args.candidate/'versus-baseline.json', {'baseline': args.baseline, 'candidate': args.candidate, 'rows': rows})
    lines = ['# Vertailu lähtöversioon', '',
             'Negatiivinen muutos tarkoittaa pienempää virhettä. Taulukko näyttää myös pahimman yksittäisen kohdan.', '',
             '| Raita | Mode | Lähtövirhe dB | Uusi virhe dB | Pahin muutos dB |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        clips = row['clips']
        if clips:
            lines.append(f"| {row['id']} {row['family']} | {row['mode']} | "
                         f"{np.mean([c['baseline_error_db'] for c in clips]):.2f} | "
                         f"{np.mean([c['candidate_error_db'] for c in clips]):.2f} | "
                         f"{max(c['error_change_db'] for c in clips):.2f} |")
    (work/args.candidate/'VERTAILU.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
