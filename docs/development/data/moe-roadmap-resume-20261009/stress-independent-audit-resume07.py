from pathlib import Path
import collections, datetime, hashlib, json, math
import numpy as np
q = Path(__file__).resolve().parent
results = json.loads((q / 'stress-results.json').read_text())
protocol = json.loads((q / 'state-protocol.json').read_text())
items = [json.loads(x) for x in (q / 'stress/cases.jsonl').read_text().splitlines()]
expected = []
for kv in protocol['kv']:
    for ub in protocol['ubatch']:
        counts = collections.Counter()
        for order in protocol['orders']:
            for variant in order:
                counts[variant] += 1
                expected.append(f'stress-{kv}-ub{ub}-{variant}-{counts[variant]}')
assert list(results) == expected[:len(results)], 'Original registered order changed'
out = {}
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save():
    summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gate': 'DESCRIPTIVE_COMPLETE' if len(out) == 36 else 'DESCRIPTIVE_PARTIAL', 'registered_processes_recorded': len(results), 'processes_audited': len(out), 'complete_return0_processes_audited': sum(x['audit'] == 'FULL_RAW_AUDITED' for x in out.values()), 'expected_processes': 36, 'groups': out, 'limits': 'Original gates retained. Failed or incomplete processes are classified without replacement. Full vectors use first historical resident-1 rep0 per prefix and KV/ubatch; an unavailable reference stays unavailable. No timing ranking or epsilon substitution. A later exact candidate does not repair baseline cycle instability.'}
    (q / 'stress-independent-audit-resume07.json').write_text(json.dumps(summary, indent=2) + '\n')
for name, result in results.items():
    folder = q / result.get('run_folder', 'stress/runs/' + name)
    response = folder / 'responses.jsonl'
    parts = name.split('-'); ub = int(parts[2][2:])
    first_name = '-'.join(parts[:3]) + '-resident-1'
    first = results[first_name]
    reference_folder = q / first.get('run_folder', 'stress/runs/' + first_name)
    record = {'original_gate': result['gate'], 'returncode': result.get('returncode'), 'stop_reason': result.get('stop_reason'), 'run_folder': str(folder.relative_to(q))}
    if result.get('returncode') != 0 or not response.exists():
        record.update(audit='INCOMPLETE_OR_FAILED_PROCESS', original_error=result.get('error'), completed_response_rows=len(response.read_text().splitlines()) if response.exists() else 0)
        out[name] = record; save(); print(name, record['audit'], record.get('original_error'), flush=True); continue
    rows = [json.loads(x) for x in response.read_text().splitlines()]
    assert len(rows) == 15, 'Return0 process has incomplete response rows'
    cases = []
    for item, row in zip(items, rows):
        assert row['id'] == item['id'] and row['prompt_ids'] == item['prefill_ids'] and row['generated_ids'] == item['decode'] and row['mode'] == 'teacher_forced'
        depth = len(row['prompt_ids']); prefill = math.ceil(depth / min(ub, 2048)); vectors = prefill + 128
        assert row['raw_vectors'] == vectors
        path = folder / (row['id'] + '-logits.bin'); size = vectors * 248320 * 4
        assert path.stat().st_size == size
        reference_path = reference_folder / (f'depth{depth}-rep0-logits.bin')
        available = reference_path.exists() and reference_path.stat().st_size == size
        a = np.memmap(reference_path, np.float32, 'r').reshape(vectors, 248320) if available else None
        b = np.memmap(path, np.float32, 'r').reshape(vectors, 248320)
        different = []; invalid = []; reference_invalid = []; first_difference = None; argmax_changes = 0
        for i in range(vectors):
            bv = b[i]
            if not np.isfinite(bv).all() or not np.any(bv): invalid.append(i)
            if a is None: continue
            av = a[i]; valid_pair = np.isfinite(av).all() and np.isfinite(bv).all() and np.any(av) and np.any(bv)
            if not np.isfinite(av).all() or not np.any(av): reference_invalid.append(i)
            if np.array_equal(av.view(np.uint32), bv.view(np.uint32)): continue
            different.append(i)
            if valid_pair: argmax_changes += int(np.argmax(av) != np.argmax(bv))
            if first_difference is None:
                phase = 'prefill' if i < prefill else 'decode'
                token_end = min(depth, (i + 1) * min(ub, 2048)) - 1 if phase == 'prefill' else depth + i - prefill
                delta = np.asarray(av, np.float64) - np.asarray(bv, np.float64) if valid_pair else None
                first_difference = {'vector_zero_based': i, 'phase': phase, 'token_end_zero_based': token_end, 'different_elements': int(np.count_nonzero(av.view(np.uint32) != bv.view(np.uint32))), 'max_abs': float(np.abs(delta).max()) if valid_pair else None, 'RMS': float(np.sqrt(np.mean(delta * delta))) if valid_pair else None, 'argmax_equal': bool(np.argmax(av) == np.argmax(bv)) if valid_pair else None}
        cases.append({'case': row['id'], 'depth': depth, 'vectors': vectors, 'full_raw_sha256': sha(path), 'reference_path': str(reference_path.relative_to(q)), 'reference_available': available, 'invalid_vectors': invalid, 'reference_invalid_vectors': reference_invalid, 'different_vectors': len(different) if available else None, 'argmax_changed_vectors': argmax_changes if available else None, 'first_difference': first_difference})
    record.update(audit='FULL_RAW_AUDITED', all_cases_complete=True, different_cases=sum(x['different_vectors'] is not None and x['different_vectors'] > 0 for x in cases), invalid_cases=sum(bool(x['invalid_vectors']) for x in cases), unavailable_reference_cases=sum(not x['reference_available'] for x in cases), cases=cases)
    out[name] = record; save(); print(name, 'different_cases', record['different_cases'], 'invalid_cases', record['invalid_cases'], 'unavailable_reference_cases', record['unavailable_reference_cases'], flush=True)
