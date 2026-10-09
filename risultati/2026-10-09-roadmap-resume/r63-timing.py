from pathlib import Path
import json, sys, shutil, collections
q = Path(__file__).resolve().parent
sys.path.insert(0, str(q))
from r63_common import r,freeze
gates=json.loads((q/'r63-model-results.json').read_text())
assert len(gates)==3 and all(x['eligible']for x in gates.values())
assert json.loads((q/'r63-raw-audit.json').read_text())['gate']=='PASS_ALL48_FULL_RAW'
results = {}
for case, item in freeze['inputs'].items():
    r.OUT = q / 'r63-timing' / case
    r.OUT.mkdir(parents=True)
    shutil.copy2(q / item['path'], r.OUT / 'tokens.csv')
    shutil.copy2(q / 'r63-model' / case / 'reference.csv', r.OUT / 'reference.csv')
    counts = collections.Counter()
    cell = {}
    for variant in ['resident', 'old', 'off', 'on', 'on', 'off', 'old', 'resident'] * 2:
        counts[variant] += 1
        name = 'timing-' + variant + '-' + str(counts[variant])
        label = 'R61-BASE' if variant == 'resident' else 'R61-OLD' if variant == 'old' else 'R61-V2'
        flags = list(r.FLAGS) + ['--load-mode', 'mmap']
        extra = {'MOE_REPLAY_WARMUP': '0'}
        if variant == 'resident':
            flags[flags.index('-ncmoe'):flags.index('-ncmoe')] = ['-ot', '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0']
        else: extra.update(GGML_SCHED_EXPERT_GPU_LAYER='17', GGML_SCHED_EXPERT_POOL='17:128:256')
        if variant == 'on': extra['GGML_SCHED_EXPERT_POOL_DECODE_ONLY'] = '1'
        try:
            assert shutil.disk_usage(q).free > 8 * 1024**3, "Disk reserve below8GiB"
            result = r.run(name, label=label, reps=1, flags=flags, extra=extra)
            assert result['metrics']['per_call_matches_reference'], 'Timing fingerprint mismatch'
            result['gate'] = 'PASS'
        except Exception as error:
            p = r.OUT / (name + '-result.json')
            result = json.loads(p.read_text()) if p.exists() else {}
            result.update(gate='FAIL', error=str(error))
        cell[name] = result
        results[case] = {'results': cell, 'inputs': item, 'eligible': len(cell) == 16 and all(x['gate'] == 'PASS' for x in cell.values())}
        (q / 'r63-timing-results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(case, name, result['gate'], result.get('error', ''), flush=True)
print('All3 preregistered timing cells complete; no silent ranking of failures', flush=True)
