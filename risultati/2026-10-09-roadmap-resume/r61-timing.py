from pathlib import Path
import json, sys, shutil, collections
q = Path(__file__).resolve().parent
sys.path.insert(0, str(q))
import guarded_runner as r
freeze = json.loads((q / 'r61-model-freeze.json').read_text())
gates = json.loads((q / 'r61-model-results.json').read_text())
assert len(gates) == 7 and all(x['eligible'] for x in gates.values())
r.CAMPAIGN = q
r.FROZEN = q / 'frozen'
r.FREEZE = freeze
def verify():
    for name, digest in freeze['artifacts'].items(): assert r.sha(q / name) == digest, name
    assert r.sha(q / 'r61-protocol.json') == freeze['protocol_sha256']
    for item in freeze['inputs'].values(): assert r.sha(q / item['path']) == item['sha256']
r.verify_freeze = verify
r.workload = lambda: r.replay_summary.read_tokens(r.OUT / 'tokens.csv')
r.REFERENCE = 'reference.csv'
results = {}
for case, item in freeze['inputs'].items():
    r.OUT = q / 'r61-timing' / case
    r.OUT.mkdir(parents=True)
    shutil.copy2(q / item['path'], r.OUT / 'tokens.csv')
    shutil.copy2(q / 'r61-model' / case / 'reference.csv', r.OUT / 'reference.csv')
    counts = collections.Counter()
    cell = {}
    for variant in ['resident', 'old', 'off', 'on', 'on', 'off', 'old', 'resident'] * 2:
        counts[variant] += 1
        name = 'timing-' + variant + '-' + str(counts[variant])
        label = 'R61-BASE' if variant == 'resident' else 'R61-OLD' if variant == 'old' else 'R61-V2'
        flags = list(r.FLAGS) + ['--load-mode', 'mmap']
        extra = {'MOE_REPLAY_WARMUP': '1'}
        if variant == 'resident':
            flags[flags.index('-ncmoe'):flags.index('-ncmoe')] = ['-ot', '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0']
        else: extra.update(GGML_SCHED_EXPERT_GPU_LAYER='17', GGML_SCHED_EXPERT_POOL='17:128:256')
        if variant == 'on': extra['GGML_SCHED_EXPERT_POOL_DECODE_ONLY'] = '1'
        try:
            result = r.run(name, label=label, reps=3, flags=flags, extra=extra)
            assert result['metrics']['per_call_matches_reference'], 'Timing fingerprint mismatch'
            result['gate'] = 'PASS'
        except Exception as error:
            p = r.OUT / (name + '-result.json')
            result = json.loads(p.read_text()) if p.exists() else {}
            result.update(gate='FAIL', error=str(error))
        cell[name] = result
        results[case] = {'results': cell, 'inputs': item, 'eligible': len(cell) == 16 and all(x['gate'] == 'PASS' for x in cell.values())}
        (q / 'r61-timing-results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(case, name, result['gate'], result.get('error', ''), flush=True)
print('All7 preregistered timing cells complete; no silent ranking of failures', flush=True)
