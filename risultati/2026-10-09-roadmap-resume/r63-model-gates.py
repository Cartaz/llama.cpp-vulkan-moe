from pathlib import Path
import json, sys, csv, shutil
import numpy as np
q = Path(__file__).resolve().parent
sys.path.insert(0, str(q))
from r63_common import r,freeze
import collections
results = {}
for case, item in freeze['inputs'].items():
    r.OUT = q / 'r63-model' / case
    r.OUT.mkdir(parents=True)
    shutil.copy2(q / item['path'], r.OUT / 'tokens.csv')
    vectors = (item['PP'] + 511) // 512 + item['TG']
    r.check_raw = lambda f: r.replay_summary.check_logits(f, vectors, 248320)
    reference = r.OUT / 'cold-gate-resident-1-logits.bin'
    warm_reference = q / 'r61-model' / case / 'gate-resident-logits.bin'
    counts = collections.Counter()
    cell = {}
    for variant in ['resident','old','off','on','on','off','old','resident'] * 2:
        counts[variant] += 1
        name = 'cold-gate-' + variant + '-' + str(counts[variant])
        label = 'R61-BASE' if variant == 'resident' else 'R61-OLD' if variant == 'old' else 'R61-V2'
        flags = list(r.FLAGS) + ['--load-mode', 'mmap']
        extra = {'MOE_REPLAY_WARMUP': '0'}
        if variant == 'resident':
            flags[flags.index('-ncmoe'):flags.index('-ncmoe')] = ['-ot', '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0']
        else:
            extra.update(GGML_SCHED_EXPERT_GPU_LAYER='17', GGML_SCHED_EXPERT_POOL='17:128:256', GGML_SCHED_EXPERT_POOL_LOG=str(r.OUT / (name + '-pool.csv')))
        if variant == 'on': extra['GGML_SCHED_EXPERT_POOL_DECODE_ONLY'] = '1'
        try:
            assert shutil.disk_usage(q).free > 8 * 1024**3, 'Disk reserve below8GiB'
            result = r.run(name, label=label, reps=1, raw=True, flags=flags, extra=extra)
            raw = r.OUT / (name + '-logits.bin')
            data = np.memmap(raw, np.uint8, 'r').reshape(1, vectors * 248320 * 4)
            assert all(np.array_equal(data[0], x) for x in data[1:]), 'Within-process raw divergence'
            warm = np.memmap(warm_reference, np.uint8, 'r')[:vectors * 248320 * 4]
            assert np.array_equal(data[0], warm), 'Cold output differs from archived warm B1'
            assert reference.exists() and reference.stat().st_size == vectors * 248320 * 4, 'Original cold B1 reference unavailable'
            if name == 'cold-gate-resident-1':
                calls = r.replay_summary.read_calls(r.OUT / (name + '.csv'))
                with (r.OUT / 'reference.csv').open('x', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=r.replay_summary.CALL_FIELDS)
                    w.writeheader()
                    w.writerows(x for x in calls if x['rep'] == 0)
            ref = np.memmap(reference, np.uint8, 'r')[:vectors * 248320 * 4]
            assert all(np.array_equal(x, ref) for x in data), 'Same-GPU B1 raw divergence'
            if variant != 'resident':
                assert result['metrics']['per_call_matches_reference'], 'Fingerprints differ from B1'
                rows = list(csv.DictReader((r.OUT / (name + '-pool.csv')).open()))
                assert len(rows) == 1 + item['TG'] and int(rows[0]['hits']) == 0, 'Unexpected warmup or nonempty first pool'
                assert any(x['reason'] == 'ready' and int(x['n_tokens']) == 1 and int(x['projections']) == 3 for x in rows), 'No admitted TG triplet'
                if variant == 'on':
                    skipped = [x for x in rows if x['reason'] == 'batched_bypass']
                    assert skipped and all(all(int(x[k]) == 0 for k in ['projections', 'hits', 'misses', 'evictions', 'upload_bytes']) for x in skipped)
                    assert not any(x['reason'] == 'ready' and int(x['n_tokens']) > 1 for x in rows)
                    result['batched_skips'] = len(skipped)
            result.update(gate='PASS', raw_vectors=vectors, raw_sha256=r.sha(raw))
        except Exception as error:
            p = r.OUT / (name + '-result.json')
            result = json.loads(p.read_text()) if p.exists() else {}
            result.update(gate='FAIL', error=str(error))
        cell[name] = result
        results[case] = {'results': cell, 'inputs': item, 'eligible': len(cell) == 16 and all(x['gate'] == 'PASS' for x in cell.values())}
        (q / 'r63-model-results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(case, name, result['gate'], result.get('error', ''), flush=True)
assert len(results) == 3 and all(x['eligible'] for x in results.values()), 'Preserve failing raw cells; timing gate stays closed'
