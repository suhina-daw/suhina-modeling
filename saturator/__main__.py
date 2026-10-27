# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Local capture inventory, native renders, measurements and listening packages."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal, optimize

from . import audio
from .project import inspect, assemble

MODES = ['Analog Clip', 'Soft Sine', 'Bass Shaper', 'Medium Curve', 'Hard Curve',
         'Sinoid Fold', 'Digital Clip', 'Waveshaper']


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def required(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'{name} is not set')
    return Path(value)


def inventory(args, work):
    data = inspect(args.project, required('SUHINA_SATURATOR_CAPTURES'))
    # Complete validation before writing any output.
    for track in data['tracks']:
        assemble(track)
    save(work/'inventory.json', data)
    print(f"Inventoried {len(data['tracks'])} tracks; export gain still needs verification.")


def render(args, work):
    data = load(work/'inventory.json')
    dest = work/args.label
    if dest.exists():
        raise FileExistsError(dest)
    probe = required('SUHINA_SATURATOR_PROBE')
    provenance = {'commit': args.commit, 'probe_sha256': audio.digest(probe),
                  'inventory_sha256': audio.digest(work/'inventory.json'), 'tracks': []}
    for track in data['tracks']:
        ident = track['id']
        dry = assemble(track)
        input_path = dest/f'{ident}-input.wav'
        output_path = dest/f'{ident}.wav'
        schedule = dest/f'{ident}.json'
        audio.write(input_path, 48000, dry)
        save(schedule, {'settings': track['settings'], 'changes': track['changes']})
        subprocess.run([str(probe), str(input_path), str(schedule), str(output_path)], check=True)
        sr, y = audio.read(output_path)
        if len(y) != len(dry):
            raise ValueError('probe changed frame count')
        provenance['tracks'].append({'id': ident, 'input_sha256': audio.digest(input_path),
                                     'output_sha256': audio.digest(output_path)})
        print('Rendered', ident, flush=True)
    save(dest/'provenance.json', provenance)


def fundamental(x, sr):
    y = x[:, 0]
    frequencies, power = signal.periodogram(y, sr, window='hann', nfft=4*len(y))
    i = int(np.argmax(power[1:]) + 1)
    guess = frequencies[i]
    t = np.arange(len(y)) / sr
    def score(hz):
        phase = 2*np.pi*hz*t
        a = np.column_stack([np.ones(len(y)), np.cos(phase), np.sin(phase)])
        coef = np.linalg.lstsq(a, y, rcond=None)[0]
        return float(np.mean((y - a@coef)**2))
    fit = optimize.minimize_scalar(score, bounds=(max(1, guess-sr/len(y)), guess+sr/len(y)), method='bounded')
    if fit.fun > np.mean(y*y) * .001:
        return None  # not a sufficiently pure stationary sine
    return float(fit.x)


def spectrum_error(a, b):
    n = min(4096, len(a))
    if n < 64:
        return None
    f, pa = signal.welch(a, 48000, axis=0, nperseg=n)
    _, pb = signal.welch(b, 48000, axis=0, nperseg=n)
    pa, pb = pa.mean(axis=1), pb.mean(axis=1)
    valid = (f >= 20) & (f <= 20000) & (pa > max(pa.max()*1e-6, 1e-18))
    return float(np.mean(np.abs(10*np.log10(np.maximum(pb[valid],1e-24)/pa[valid])))) if valid.any() else None


def compare(args, work):
    inventory = load(work/'inventory.json')
    results = []
    for track in inventory['tracks']:
        if audio.digest(track['reference']) != track['sha256']:
            raise ValueError('reference changed after inventory')
        _, ref = audio.read(track['reference'])
        _, candidate = audio.read(work/args.label/f"{track['id']}.wav")
        _, dry = audio.read(work/args.label/f"{track['id']}-input.wav")
        if ref.shape != candidate.shape or dry.shape != ref.shape:
            raise ValueError('length mismatch')
        # Use the first input-active clip at the first mode for a single track delay.
        first = track['clips'][0]
        lo, hi = first['start'], first['end']
        lag = audio.delay(ref[lo:hi], candidate[lo:hi])
        aligned = audio.advance(candidate, lag)
        fullmask = audio.active_mask(dry)
        rows = []
        for i, change in enumerate(track['changes']):
            start = change['frame']
            end = track['changes'][i+1]['frame'] if i+1 < len(track['changes']) else len(ref)
            mode_row = {'mode': MODES[int(change['value'])], 'start_seconds': start/48000,
                        'end_seconds': end/48000, 'clips': []}
            for clip_index, clip in enumerate(track['clips']):
                lo, hi = max(start+4800, clip['start']), min(end, clip['end'])
                if hi <= lo:
                    continue
                mask = fullmask[lo:hi]
                row = audio.metrics(ref[lo:hi], aligned[lo:hi], mask)
                row.update({'clip': clip_index, 'start_frame': lo, 'end_frame': hi})
                row['input_rms_dbfs'] = audio.db(audio.rms(dry[lo:hi][mask]))
                row['input_peak'] = float(np.max(np.abs(dry[lo:hi])))
                row['spectral_error_db'] = spectrum_error(ref[lo:hi], aligned[lo:hi])
                if track['family'] in ['sine steps', 'sine']:
                    # Fit steady inner portion; never interpret transients as THD.
                    a, b = lo+4800, min(hi-2400, lo+28800)
                    if b-a >= 4800 and fullmask[a:b].all():
                        freq = fundamental(dry[a:b], 48000)
                        if freq is not None:
                            row['reference_harmonics'] = audio.harmonics(ref[a:b], 48000, freq)
                            row['candidate_harmonics'] = audio.harmonics(aligned[a:b], 48000, freq)
                mode_row['clips'].append(row)
            transition_end = min(end, start+4800)
            mode_row['transition'] = audio.metrics(ref[start:transition_end], aligned[start:transition_end],
                                                   fullmask[start:transition_end])
            rows.append(mode_row)
        quiet = ~fullmask
        results.append({'id': track['id'], 'family': track['family'], 'holdout': track['holdout'],
                        'candidate_delay_samples': lag, 'modes': rows,
                        'quiet_reference_rms_dbfs': audio.db(audio.rms(ref[quiet])),
                        'quiet_candidate_rms_dbfs': audio.db(audio.rms(candidate[quiet]))})
        print('Compared', track['id'], flush=True)
    report = {'label': args.label, 'provenance': load(work/args.label/'provenance.json'),
              'export_gain_verified': inventory['export_gain_verified'], 'tracks': results,
              'limits': ['No parameter fitting until input and export calibration are verified.',
                         'One common fixed delay per track; no gain normalization.',
                         'No aliasing claim from arbitrary musical sources.']}
    save(work/args.label/'comparison.json', report)
    lines = ['# Saturator comparison', '', f'Version: {args.label}', '',
             '| Track | Mode | Mean gain error dB | Mean spectral error dB |', '|---|---|---:|---:|']
    for track in results:
        for mode in track['modes']:
            clips = [c for c in mode['clips'] if c['status'] == 'measured']
            if clips:
                gain = np.mean([c['gain_error_db'] for c in clips])
                spectra = [c['spectral_error_db'] for c in clips if c['spectral_error_db'] is not None]
                spec = f'{np.mean(spectra):.3f}' if spectra else 'unavailable'
                lines.append(f"| {track['id']} | {mode['mode']} | {gain:.3f} | {spec} |")
    (work/args.label/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def listen(args, work):
    inventory = load(work/'inventory.json')
    dest = work/args.name
    if dest.exists():
        raise FileExistsError(dest)
    manifest = []
    for track in inventory['tracks']:
        _, ref = audio.read(track['reference'])
        versions = [('A-reference', ref)]
        for letter, label in zip('BCDE', args.labels):
            _, x = audio.read(work/label/f"{track['id']}.wav")
            versions.append((f'{letter}-{label}', x))
        _, dry = audio.read(work/args.labels[0]/f"{track['id']}-input.wav")
        mask = audio.active_mask(dry)
        for i, change in enumerate(track['changes']):
            start = change['frame']
            end = track['changes'][i+1]['frame'] if i+1<8 else len(ref)
            active = np.flatnonzero(mask[start:end])
            if not len(active):
                continue
            lo, hi = start+int(active[0]), min(end, start+int(active[-1])+2401)
            clips = [(label, x[lo:hi]) for label, x in versions]
            target = min(audio.rms(x) for _, x in clips)
            gains = [target/max(audio.rms(x), 1e-15) for _, x in clips]
            peak = max(float(np.max(np.abs(signal.resample_poly(x*g, 4, 1, axis=0))))
                       for (_, x), g in zip(clips, gains))
            common = min(1, 10**(-1/20)/max(peak, 1e-15))
            for (label, x), gain in zip(clips, gains):
                for kind, g in [('original', 1.0), ('matched', gain*common)]:
                    path = dest/kind/f"{track['id']}-{i+1}-{label}.wav"
                    audio.write(path, 48000, x*g)
                    manifest.append({'file': str(path.relative_to(dest)), 'start_frame': lo,
                                     'end_frame': hi, 'gain_db': audio.db(g), 'sha256': audio.digest(path)})
    save(dest/'manifest.json', manifest)
    guide = (
        'A = referenssi. B, C jne. = nimetyt Suhina-versiot.\n'
        'Ensimmäinen numero on raita, toinen mode 1–8.\n'
        'original: alkuperäiset tasot, aloita hiljaisella kuunteluvoimakkuudella.\n'
        'matched: vakiovahvistuksella tasatut versiot, ei limitteriä.\n'
        'Sama aikaväli kaikilla versioilla. Kiinteää viivettä ei poisteta kuuntelusta.\n\n'
        'Modet:\n' + ''.join(f'{i+1}: {name}\n' for i, name in enumerate(MODES)) +
        '\nRaidat:\n' + ''.join(f"{t['id']}: {t['family']}\n" for t in inventory['tracks']))
    (dest/'00 - ALOITA TASTA.txt').write_text(guide, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory'); p.add_argument('--project', required=True)
    p = sub.add_parser('render'); p.add_argument('--label', required=True); p.add_argument('--commit', required=True)
    p = sub.add_parser('compare'); p.add_argument('--label', required=True)
    p = sub.add_parser('listen'); p.add_argument('--name', required=True); p.add_argument('labels', nargs='+')
    args = parser.parse_args()
    labels = [getattr(args, field, None) for field in ['label', 'name']]
    labels.extend(getattr(args, 'labels', []))
    if len(getattr(args, 'labels', [])) > 4:
        parser.error('listening packages support at most four candidate versions')
    for label in labels:
        if label and (not label.replace('-', '').replace('_', '').isalnum()):
            parser.error('labels must contain only letters, numbers, hyphens and underscores')
    work = Path(os.environ.get('SUHINA_SATURATOR_WORK', 'target/saturator'))
    try:
        globals()[args.command](args, work)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
