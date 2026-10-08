#!/usr/bin/env python3
"""Validate and extract the RX 6800 layer 17 CPU observer's little-endian captures."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

MAGIC = 0x31454F4D55504347
SHAPES = [(2048, 512, 1), (2048, 512, 1), (512, 2048, 8)]
NAMES = ['gate', 'up', 'down']


def floats(raw, rows, cols, label):
    for row in range(rows):
        values = struct.unpack_from('<' + str(cols) + 'f', raw, row * cols * 4)
        if not all(math.isfinite(v) for v in values) or not any(v != 0 for v in values):
            raise ValueError(label + ': invalid float vector')


def read_capture(path, allow_different_inputs=False):
    raw = Path(path).read_bytes()
    offset = 0
    records = []
    def take(size):
        nonlocal offset
        if len(raw) - offset < size:
            raise ValueError('truncated capture at byte ' + str(offset))
        part = raw[offset:offset + size]
        offset += size
        return part
    while offset < len(raw):
        magic, version, projection, k, m, slots, nids, experts = struct.unpack('<8Q', take(64))
        if magic != MAGIC or version not in (1, 2) or projection >= 3 or (nids, experts) != (8, 256):
            raise ValueError('invalid capture header')
        if (k, m, slots) != SHAPES[projection]:
            raise ValueError('unexpected layer 17 shape')
        row_bytes = 0
        if version == 2:
            qtype, row_bytes = struct.unpack('<2Q', take(16))
            if qtype != 15 or row_bytes != k // 256 * 292:
                raise ValueError('invalid Q8_K layout')
        original = take(k * slots * 4)
        ids = take(32)
        floats(original, slots, k, 'input')
        if any(v < 0 or v >= 256 for v in struct.unpack('<8i', ids)):
            raise ValueError('invalid routing ID')
        quantized = take(row_bytes * slots) if version == 2 else b''
        for pos in range(0, len(quantized), 292):
            scale = struct.unpack_from('<f', quantized, pos)[0]
            qs = struct.unpack_from('<256b', quantized, pos + 4)
            sums = struct.unpack_from('<16h', quantized, pos + 260)
            if not math.isfinite(scale) or any(sum(qs[i*16:(i+1)*16]) != sums[i] for i in range(16)):
                raise ValueError('invalid Q8_K scale or block sums')
        output = take(m * 8 * 4) if version == 2 else b''
        if output:
            floats(output, 8, m, 'output')
        records.append({'version':version, 'projection':projection, 'original':original,
                        'ids':ids, 'quantized':quantized, 'output':output})
    if not records or len(records) % 3 or len({r['version'] for r in records}) != 1:
        raise ValueError('capture needs complete gate/up/down triplets of one version')
    for start in range(0, len(records), 3):
        group = records[start:start + 3]
        if [r['projection'] for r in group] != [0, 1, 2] or len({r['ids'] for r in group}) != 1:
            raise ValueError('inconsistent per-token projection order or routing')
        if not allow_different_inputs and group[0]['original'] != group[1]['original']:
            raise ValueError('gate/up input differs')
    return records


def extract(records, steps, directory):
    files = {}
    for index, step in enumerate(steps):
        if step < 1 or step > len(records) // 3:
            raise ValueError('decode step outside capture')
        for projection, name in enumerate(NAMES):
            record = records[3 * (step - 1) + projection]
            for key in ['original', 'ids', 'quantized', 'output']:
                if record[key]:
                    path = Path(directory) / ('real-' + name + '-' + str(index) + '-' + key + '.bin')
                    files[path] = record[key]
    if any(path.exists() for path in files):
        raise ValueError('extraction would overwrite a file')
    Path(directory).mkdir(parents=True, exist_ok=True)
    for path, data in files.items():
        with path.open('xb') as f:
            f.write(data)
    return {str(p):hashlib.sha256(data).hexdigest() for p, data in files.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 100, 200])
    parser.add_argument('--extract', type=Path)
    parser.add_argument('--allow-different-inputs', action='store_true', help='Allow intentional gate/up input interventions; routing and layout checks still apply')
    args = parser.parse_args()
    try:
        records = read_capture(args.capture, args.allow_different_inputs)
        summary = {'version':records[0]['version'], 'decode_steps':len(records)//3, 'records':len(records),
                   'input_vectors':sum(SHAPES[r['projection']][2] for r in records),
                   'output_vectors':len(records)*8 if records[0]['version']==2 else 0,
                   'sha256':hashlib.sha256(args.capture.read_bytes()).hexdigest()}
        if args.extract:
            summary['files'] = extract(records, args.steps, args.extract)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
