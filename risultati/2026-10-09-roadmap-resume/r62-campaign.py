from pathlib import Path
import json, sys
import numpy as np
q = Path(__file__).resolve().parent
sys.path.insert(0, str(q))
import guarded_runner as r
freeze = json.loads((q / 'r62-freeze.json').read_text())
protocol = json.loads((q / 'r62-protocol.json').read_text())
items = [json.loads(x) for x in (q / 'r62-cases.jsonl').read_text().splitlines()]
r.CAMPAIGN = q
r.FROZEN = q / 'frozen'
r.FREEZE = freeze
def verify():
    for name, digest in freeze['artifacts'].items(): assert r.sha(q / name) == digest, name
    assert r.sha(q / 'r62-protocol.json') == freeze['protocol_sha256']
    assert r.sha(q / 'r62-cases.jsonl') == freeze['cases_sha256']
r.verify_freeze = verify
results = {}
references = {}
for name in protocol['order']:
    placement, prefill, occurrence = name.split('-')
    group = placement + '-' + prefill
    r.OUT = q / 'r62-native' / name
    r.OUT.mkdir(parents=True)
    flags = list(r.FLAGS)
    flags[flags.index('-c') + 1] = '8192'
    flags[flags.index('-b') + 1] = '2048'
    flags += ['--load-mode', 'mmap', '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0', '--seed', '42', '-n', '64']
    if placement == 'gpu': flags[flags.index('-ncmoe'):flags.index('-ncmoe')] = ['-ot', '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0']
    extra = {'QUALITY_CASES': str(q / 'r62-cases.jsonl'), 'QUALITY_OUT': str(r.OUT), 'QUALITY_THINKING': '1', 'QUALITY_CONTINUE_ON_INVALID': '1'}
    if prefill == 'legacy': extra['QUALITY_LEGACY_PHYSICAL_PREFILL'] = '1'
    try:
        result = r.run(name, label='R62-BASE', flags=flags, extra=extra, mode='corpus')
        rows = [json.loads(x) for x in (r.OUT / 'responses.jsonl').read_text().splitlines()]
        assert len(rows) == 8
        result['cases'] = {}
        by_id = {}
        for item, row in zip(items, rows):
            assert row['id'] == item['id'] and row['seed'] == 42 and row['temperature'] == float(np.float32(.6)) and row['enable_thinking']
            assert row['prefill_protocol'] == ('legacy_physical' if prefill == 'legacy' else 'application_logical')
            path = r.OUT / row['raw_logits_file']
            assert path.stat().st_size == row['raw_vectors'] * 248320 * 4
            record = {'numerically_valid': row['numerically_valid'], 'failure_reason': row['failure_reason'], 'raw_vectors': row['raw_vectors'], 'generated_ids': len(row['generated_token_ids']), 'raw_sha256': r.sha(path), 'prompt_ids': row['prompt_token_ids'], 'raw_path': str(path.relative_to(q)), 'EOG': row['eog'], 'truncated_at64': row['truncated']}
            if row['numerically_valid']:
                r.replay_summary.check_logits(path, row['raw_vectors'], 248320)
                assert row['raw_vectors'] == len(row['generated_token_ids'])
            if item.get('repeat_of'):
                previous, previous_path = by_id[item['repeat_of']]
                record['duplicate_exact'] = row['prompt_token_ids'] == previous['prompt_token_ids'] and row['generated_token_ids'] == previous['generated_token_ids'] and path.read_bytes() == previous_path.read_bytes()
            key = (group, item['id'])
            if key in references:
                previous, previous_path = references[key]
                record['fresh_process_repeat_exact'] = row['prompt_token_ids'] == previous['prompt_token_ids'] and row['generated_token_ids'] == previous['generated_token_ids'] and path.read_bytes() == previous_path.read_bytes()
            else: references[key] = (row, path)
            by_id[item['id']] = (row, path)
            result['cases'][item['id']] = record
        result['gate'] = 'PASS_BOUNDED_DIAGNOSTIC' if all(x['numerically_valid'] and x.get('duplicate_exact', True) and x.get('fresh_process_repeat_exact', True) for x in result['cases'].values()) else 'FAIL_NUMERICAL_OR_REPEAT'
    except Exception as error:
        p = r.OUT / (name + '-result.json')
        result = json.loads(p.read_text()) if p.exists() else {}
        result.update(gate='FAIL_HARNESS_OR_RESOURCE', error=str(error))
    results[name] = result
    (q / 'r62-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(name, result['gate'], result.get('error', ''), flush=True)
print('R62 diagnostics finished;64-token cap does not qualify answer quality', flush=True)
