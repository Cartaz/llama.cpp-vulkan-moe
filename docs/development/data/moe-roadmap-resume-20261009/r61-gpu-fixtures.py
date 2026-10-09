from pathlib import Path
import json, sys, csv, hashlib
q = Path(__file__).resolve().parent
sys.path.insert(0, str(q))
import guarded_runner as r
freeze = json.loads((q / 'stress-freeze.json').read_text())
v2 = json.loads((q / 'r61-v2-freeze.json').read_text())
freeze['artifacts'].update(v2['artifacts'])
freeze['variants']['R61-V2'] = '2e08b57 + r61-engine-v2.patch; exact source hashes in r61-v2-freeze.json'
freeze['helper_source_sha256'] = v2['source']['examples/moe-trace/expert-pool-check.cpp']
r.CAMPAIGN = q
r.FROZEN = q / 'frozen'
r.FREEZE = freeze
def verify():
    for name, digest in freeze['artifacts'].items(): assert r.sha(q / name) == digest, name
    assert r.sha(q / 'r61-protocol.json') == v2['protocol_sha256']
r.verify_freeze = verify
results = {}
references = {}
for fixture in ['scheduler-pool', 'multi-pool']:
    for label, validation in [('FIXED', False), ('R61-V2', False), ('R61-V2', True)]:
        for enabled in [False, True]:
            name = fixture + '-' + label + ('-validation' if validation else '') + ('-ON' if enabled else '-OFF')
            r.OUT = q / 'r61-gpu' / name
            r.OUT.mkdir(parents=True)
            extra = {'GGML_SCHED_EXPERT_POOL_LOG': str(r.OUT / 'pool.csv')}
            if enabled: extra['GGML_SCHED_EXPERT_POOL_DECODE_ONLY'] = '1'
            if validation:
                extra.update(VK_LAYER_PATH=str(q.parent / '2026-10-08-milestones-0-3/validation-layer/extracted/usr/share/vulkan/explicit_layer.d'), VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation', VK_VALIDATION_VALIDATE_SYNC='1', VK_VALIDATION_LOG_FILENAME=str(r.OUT / 'validation.log'))
            flags = ['--vulkan', '--' + fixture, '--output-bin', str(r.OUT / 'raw.bin')]
            try:
                result = r.run(name, label=label, flags=flags, extra=extra, mode='check')
                raw = (r.OUT / 'raw.bin').read_bytes()
                assert raw
                if fixture not in references: references[fixture] = raw
                assert raw == references[fixture], 'Fixture raw differs from original candidate'
                rows = list(csv.DictReader((r.OUT / 'pool.csv').open()))
                assert any(x['reason'] == 'ready' and int(x['n_tokens']) == 1 and int(x['projections']) == 3 for x in rows), 'Single-token pool never admitted'
                if enabled and label == 'R61-V2':
                    skipped = [x for x in rows if x['reason'] == 'batched_bypass']
                    assert skipped and not any(x['reason'] == 'ready' and int(x['n_tokens']) > 1 for x in rows)
                    assert all(int(x['n_tokens']) > 1 and all(int(x[k]) == 0 for k in ['projections', 'hits', 'misses', 'evictions', 'upload_bytes']) for x in skipped)
                    result['batched_skips'] = len(skipped)
                else:
                    assert any(x['reason'] == 'ready' and int(x['n_tokens']) > 1 for x in rows), 'Original batched admission absent'
                if validation:
                    assert any('libVkLayer_khronos_validation' in p for p in result['loaded_libraries']), 'Validation layer not loaded'
                    text = (r.OUT / 'validation.log').read_text()
                    assert not text.strip(), 'Reported validation messages preserved'
                    log = (r.OUT / (name + '.log')).read_text()
                    stdout = (r.OUT / (name + '.csv')).read_text()
                    combined = log + '\n' + stdout + '\n' + text
                    assert not any(marker in combined for marker in ['Validation Error', 'VUID-', 'SYNC-HAZARD']), 'Validation reports in either output stream'
                result.update(gate='PASS', raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw))
            except Exception as error:
                p = r.OUT / (name + '-result.json')
                result = json.loads(p.read_text()) if p.exists() else {}
                result.update(gate='FAIL', error=str(error))
            results[name] = result
            (q / 'r61-gpu-results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(name, result['gate'], result.get('error', ''), flush=True)
assert len(results) == 12 and all(x['gate'] == 'PASS' for x in results.values()), 'Keep all failed fixtures; do not launch model/timing on failing mechanism gate'
