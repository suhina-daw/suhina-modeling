# Copyright (C) 2026 The Suhina contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read a self-authored .als capture project; reject unsupported signal transforms."""
import gzip
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .audio import digest, read

PARAMS = {'PreDrive': 'drive', 'PostDrive': 'output', 'DryWet': 'mix',
          'BassShaperThreshold': 'threshold', 'PostClip': 'postClip',
          'ColorOn': 'color', 'PreDcFilter': 'dcFilter', 'WsDrive': 'shaperDrive',
          'WsCurve': 'shaperCurve', 'WsDepth': 'shaperDepth', 'WsLin': 'shaperLinear',
          'WsDamp': 'shaperDamp', 'WsPeriod': 'shaperPeriod',
          'ColorFrequency': 'colorFrequency', 'ColorWidth': 'colorWidth'}


def value(node, path):
    if node is None:
        raise ValueError(f'missing project parent for: {path}')
    element = node.find(path)
    if element is None or 'Value' not in element.attrib:
        raise ValueError(f'missing project value: {path}')
    return element.get('Value')


def number(node, path):
    v = value(node, path)
    result = float({'true': '1', 'false': '0'}.get(v, v))
    if not math.isfinite(result):
        raise ValueError(f'non-finite project value: {path}')
    return result


def key(name):
    name = re.sub(r'^\d+ ', '', name)
    return re.split(r'(?:-dry| dry|-48k| 48k)', name)[0]


def type_changes(events, bpm, sr=48000):
    """Last event at a frame wins, including the negative-time initial value."""
    by_frame = {}
    for beat, mode in events:
        if not math.isfinite(beat) or not math.isfinite(mode) or mode != int(mode) or not 0 <= mode <= 7:
            raise ValueError('invalid Type automation event')
        by_frame[max(0, round(beat*60/bpm*sr))] = int(mode)
    changes = []
    for frame, mode in sorted(by_frame.items()):
        if not changes or mode != changes[-1]['value']:
            changes.append({'frame': frame, 'param': 'type', 'value': mode})
    if [e['value'] for e in changes] != list(range(8)) or changes[0]['frame'] != 0:
        raise ValueError('expected ordered eight-mode capture starting at frame zero')
    return changes


def automation_events(track, target):
    events = []
    for envelope in track.findall('.//AutomationEnvelope'):
        points = envelope.findall('Automation/Events/*')
        pointer = envelope.find('EnvelopeTarget/PointeeId')
        if not points:
            continue
        if pointer is None or pointer.get('Value') != target:
            raise ValueError('automation other than Type requires an explicit dry export or supported schedule')
        events.extend((float(e.get('Time')), float(e.get('Value'))) for e in points)
    return events


def inspect(project, captures):
    with gzip.open(project, 'rb') as stream:
        data = stream.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024 or b'<!DOCTYPE' in data:
        raise ValueError('unsupported project size or document type')
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f'invalid capture project XML: {error}') from error
    tempo = root.find('.//MainTrack/DeviceChain/Mixer/Tempo')
    if tempo is None:
        tempo = root.find('.//MasterTrack/DeviceChain/Mixer/Tempo')
    bpm = number(tempo, 'Manual')
    if bpm != 120:
        raise ValueError(f'expected confirmed 120 BPM, found {bpm}')
    files = {}
    for path in Path(captures).glob('*.wav'):
        k = key(path.stem)
        if k in files:
            raise ValueError(f'ambiguous capture key: {k}')
        files[k] = path
    dry_by_hash = {}
    for path in (Path(captures) / 'dry').glob('*.wav'):
        dry_by_hash.setdefault(digest(path), []).append(path)
    records = []
    ignored = []
    source_hashes = {}
    for index, track in enumerate(root.findall('.//Tracks/AudioTrack')):
        name = value(track, 'Name/EffectiveName')
        if not track.findall('.//AudioClip'):
            ignored.append({'track': index+1, 'reason': 'no audio clips'})
            continue
        k = key(name)
        if k not in files:
            raise ValueError(f'no capture for track {index+1}')
        devices = list(track.findall('DeviceChain/DeviceChain/Devices/*'))
        if len(devices) != 1 or devices[0].tag != 'Saturator':
            raise ValueError(f'track {index+1}: expected exactly one saturator')
        device = devices[0]
        if value(device, 'On/Manual') != 'true':
            raise ValueError('bypassed capture device')
        settings = {target: number(device, source+'/Manual') for source, target in PARAMS.items()}
        settings['colorLow'] = number(device, 'BaseDrive/Manual') / 36
        settings['colorHigh'] = number(device, 'ColorDepth/Manual') / 24
        if value(device, 'Oversampling') != 'true':
            raise ValueError('capture has high-quality mode off; not supported by baseline')
        mixer = track.find('DeviceChain/Mixer')
        if number(mixer, 'Pan/Manual') != 0 or number(mixer, 'Volume/Manual') != 1:
            raise ValueError('non-unity track mixer requires explicit calibration')
        target = device.find('Type/AutomationTarget').get('Id')
        changes = type_changes(automation_events(track, target), bpm)
        sr, reference = read(files[k])
        clips = []
        for clip in track.findall('.//AudioClip'):
            for field in ['PitchCoarse', 'PitchFine']:
                if number(clip, field) != 0:
                    raise ValueError('transposed source needs a dry arrangement export')
            for field in ['IsWarped', 'Loop/LoopOn']:
                if value(clip, field) != 'false':
                    raise ValueError('warped or looped source needs a dry arrangement export')
            if any(number(clip, 'Fades/'+field) != 0 for field in ['FadeInLength', 'FadeOutLength']):
                raise ValueError('clip fades need a dry arrangement export')
            source = Path(value(clip, './/FileRef/Path'))
            if source not in source_hashes:
                source_hashes[source] = digest(source)
            source_hash = source_hashes[source]
            matches = dry_by_hash.get(source_hash, [])
            if len(matches) > 1:
                raise ValueError(f'track {index+1}: duplicate dry copies')
            # The project is authoritative when its original still exists. Record the
            # fallback explicitly; never substitute a similarly named, different sound.
            chosen = matches[0] if matches else source
            start = round(number(clip, 'CurrentStart') * 60/bpm * sr)
            end = round(number(clip, 'CurrentEnd') * 60/bpm * sr)
            # For unwarped clips these source positions are seconds, not beats.
            offset = round((number(clip, 'Loop/LoopStart') + number(clip, 'Loop/StartRelative')) * sr)
            clips.append({'start': start, 'end': end, 'offset': offset,
                          'gain': number(clip, 'SampleVolume'), 'source': str(chosen),
                          'source_from_project': not bool(matches),
                          'sha256': source_hash})
        records.append({'id': f'{index+1:02}', 'family': k,
                        'holdout': k not in ['sine steps', 'square steps', 'triangle steps', 'saw steps'],
                        'reference': str(files[k]), 'sha256': digest(files[k]),
                        'frames': len(reference), 'sample_rate': sr, 'settings': settings,
                        'changes': changes, 'clips': clips})
    main = root.find('.//MainTrack/DeviceChain/Mixer')
    return {'project_sha256': digest(project), 'bpm': bpm, 'settings_verified_from_project': True,
            'main_gain': number(main, 'Volume/Manual') if main is not None else None,
            'export_gain_verified': False, 'tracks': records, 'ignored_tracks': ignored}


def assemble(track):
    out = np.zeros((track['frames'], 2))
    occupied = np.zeros(track['frames'], dtype=bool)
    cache = {}
    for clip in track['clips']:
        path = clip['source']
        if path not in cache:
            if digest(path) != clip['sha256']:
                raise ValueError('source changed after inventory')
            cache[path] = read(path)[1]
        x = cache[path]
        start, end, offset = clip['start'], clip['end'], clip['offset']
        if start < 0 or end <= start or end > len(out) or offset < 0 or offset+end-start > len(x):
            raise ValueError('clip outside source or capture bounds')
        if occupied[start:end].any():
            raise ValueError('overlapping clips need an explicit dry export')
        out[start:end] = x[offset:offset+end-start] * clip['gain']
        occupied[start:end] = True
    return out
