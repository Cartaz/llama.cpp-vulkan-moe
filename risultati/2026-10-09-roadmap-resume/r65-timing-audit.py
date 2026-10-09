from pathlib import Path
import csv, datetime, hashlib, json
import numpy as np

q = Path(__file__).resolve().parent
data = json.loads((q / 'r65-timing-results.json').read_text())
analysis = json.loads((q / 'r65-analysis.json').read_text())
raw = json.loads((q / 'r65-raw-audit.json').read_text())
freeze = json.loads((q / 'r65-freeze.json').read_text())
rng = np.random.default_rng(65)
errors, records = [], {}
count = expected = 0
keys = ['phase', 'position', 'n_tokens', 'logits_hash']

def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def verdict(lo, hi, margin, latency):
    if lo >= -margin and hi <= margin:
        return 'NESSUN_CAMBIAMENTO'
    if (latency and hi < -margin) or (not latency and lo > margin):
        return 'MIGLIORAMENTO'
    if (latency and lo > margin) or (not latency and hi < -margin):
        return 'REGRESSIONE'
    return 'INCONCLUDENTE'

assert set(data) == set(analysis['cells']) == {'heldout-planning', 'short'}
for case, cell in data.items():
    try:
        item = freeze['inputs'][case]
        assert cell['inputs'] == item and item['timing_allowed']
        a = analysis['cells'][case]
        if not raw['cells'][case]['eligible']:
            assert not cell['eligible'] and not cell['results'] and a['gate'] == 'INELIGIBLE'
            records[case] = {'gate': 'INELIGIBLE_RAW', 'processes': 0}
            continue
        expected += 8
        assert cell['eligible'] and len(cell['results']) == 8
        folder = q / 'r65-timing' / case
        original = q / 'r65-model' / case / 'raw-B1-1.csv'
        ref = [tuple(row[k] for k in keys) for row in read_csv(original) if int(row['rep']) == 0]
        assert [tuple(row[k] for k in keys) for row in read_csv(folder / 'reference.csv')] == ref
        assert hashlib.sha256((folder / 'tokens.csv').read_bytes()).hexdigest() == item['sha256']
        values = {'B1': [], 'B4': []}
        times, hashes = [], {}
        for variant in values:
            for occurrence in range(1, 5):
                name = f'timing-{variant}-{occurrence}'
                manifest = json.loads((folder / (name + '-result.json')).read_text())
                assert manifest == {k: v for k, v in cell['results'][name].items() if k != 'gate'}
                env = manifest['environment']
                assert manifest['returncode'] == 0 and env['MOE_REPLAY_REPS'] == '3' and env['MOE_REPLAY_WARMUP'] == '1'
                assert 'MOE_REPLAY_LOGITS_OUT' not in env and not any(k.startswith('GGML_SCHED_') for k in env)
                assert len(manifest['expected_libraries']) == 6
                assert all(manifest['loaded_libraries'].get(p) == digest for p, digest in manifest['expected_libraries'].items())
                assert manifest['label'] == 'R65-' + variant
                assert manifest['source_commit'] == freeze['variants']['R65-' + variant]
                assert manifest['argv'][0] == str(q / 'frozen' / ('R65-' + variant) / 'replay')
                times.append((manifest['started_utc'], variant))
                path = folder / (name + '.csv')
                rows = read_csv(path)
                assert len(rows) == 3 * len(ref)
                for rep in range(3):
                    assert [tuple(row[k] for k in keys) for row in rows if int(row['rep']) == rep] == ref
                pp = sum(int(row['elapsed_us']) for row in rows if row['phase'] == 'prefill') / 1000
                tg = sum(int(row['elapsed_us']) for row in rows if row['phase'] == 'decode') / 1000
                assert pp > 0 and tg > 0
                values[variant].append({'pp_tps': item['PP'] * 3000 / pp, 'tg_tps': item['TG'] * 3000 / tg, 'replay_decode_ms': (pp + tg) / 3, 'tg_p95_ms': float(np.quantile([int(row['elapsed_us']) / 1000 for row in rows if row['phase'] == 'decode'], .95))})
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                count += 1
        assert [v for _, v in sorted(times)] == ['B1', 'B4', 'B4', 'B1'] * 2
        drift = {k: (values['B1'][-1][k] / values['B1'][0][k] - 1) * 100 for k in values['B1'][0]}
        assert all(np.isclose(a['drift_percent'][k], v, rtol=1e-12, atol=1e-12) for k, v in drift.items())
        for variant in values:
            for actual, saved in zip(values[variant], a['process_values'][variant]):
                assert all(np.isclose(actual[k], saved[k], rtol=1e-12, atol=1e-12) for k in actual)
        records[case] = {'process_values': values, 'csv_sha256': hashes, 'drift_percent': drift}
        if any(abs(v) > 20 for v in drift.values()):
            assert a['gate'] == 'INELIGIBLE'
            continue
        assert a['gate'] == 'ELIGIBLE_BOUNDED_REPLAY'
        for metric in values['B1'][0]:
            aa = np.array([v[metric] for v in values['B1']])
            bb = np.array([v[metric] for v in values['B4']])
            samples = (np.median(bb[rng.integers(0, 4, (50000, 4))], axis=1) / np.median(aa[rng.integers(0, 4, (50000, 4))], axis=1) - 1) * 100
            ci = np.quantile(samples, [.025, .975])
            saved = a['B4_vs_B1'][metric]
            assert np.isclose((np.median(bb) / np.median(aa) - 1) * 100, saved['effect_percent'], rtol=1e-12, atol=1e-12)
            assert np.allclose(ci, saved['bootstrap95_percent'], rtol=1e-10, atol=1e-10)
            assert saved['verdict'] == verdict(*ci, 5 if metric == 'tg_p95_ms' else 3, metric.endswith('_ms'))
    except Exception as error:
        errors.append({'case': case, 'error': repr(error)})
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gate': 'PASS_COMPLETE_CSV_CI_AUDIT' if not errors and count == expected else 'FAIL_AUDIT', 'processes': count, 'expected': expected, 'errors': errors, 'cells': records, 'limits': 'Independent full CSV/order/fingerprint/DSO/environment, PP/TG sums,p95,drift and bootstrap/verdict audit. Ineligible raw cells remain in the report. No long timing or general performance qualification.'}
(q / 'r65-timing-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['gate'], count, expected, flush=True)
assert not errors and count == expected
