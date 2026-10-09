from pathlib import Path
import json, sys, datetime
import numpy as np
q = Path(__file__).resolve().parent
sys.path.insert(0, str(q))
import guarded_runner as r
data = json.loads((q / 'r61-timing-results.json').read_text())
rng = np.random.default_rng(61)
summary = {}
def verdict(lo, hi, margin, latency):
    if lo >= -margin and hi <= margin: return 'NESSUN_CAMBIAMENTO'
    if latency:
        if hi < -margin: return 'MIGLIORAMENTO'
        if lo > margin: return 'REGRESSIONE'
    else:
        if lo > margin: return 'MIGLIORAMENTO'
        if hi < -margin: return 'REGRESSIONE'
    return 'INCONCLUDENTE'
for case, cell in data.items():
    if not cell['eligible']:
        summary[case] = {'gate': 'INELIGIBLE', 'reason': 'Incomplete or failed timing gate'}
        continue
    values = {}
    for variant in ['resident', 'old', 'off', 'on']:
        records = []
        for i in range(1, 5):
            name = 'timing-' + variant + '-' + str(i)
            result = cell['results'][name]
            reps = result['metrics']['repetitions']
            pp_ms = sum(x['pp_elapsed_ms'] for x in reps)
            tg_ms = sum(x['tg_elapsed_ms'] for x in reps)
            calls = r.replay_summary.read_calls(q / 'r61-timing' / case / (name + '.csv'))
            records.append({'pp_tps': cell['inputs']['PP'] * len(reps) * 1000 / pp_ms, 'tg_tps': cell['inputs']['TG'] * len(reps) * 1000 / tg_ms, 'replay_decode_ms': (pp_ms + tg_ms) / len(reps), 'tg_p95_ms': r.replay_summary.percentile([x['elapsed_us'] / 1000 for x in calls if x['phase'] == 'decode'], .95)})
        values[variant] = records
    drift = {k: (values['resident'][-1][k] / values['resident'][0][k] - 1) * 100 for k in values['resident'][0]}
    if any(abs(x) > 20 for x in drift.values()):
        summary[case] = {'gate': 'INELIGIBLE', 'reason': 'Resident process drift exceeds20%', 'drift_percent': drift, 'process_values': values}
        continue
    comparisons = {}
    for base, candidate in [('old', 'off'), ('off', 'on'), ('old', 'on'), ('resident', 'on')]:
        metrics = {}
        for metric in values[base][0]:
            a = np.array([x[metric] for x in values[base]])
            b = np.array([x[metric] for x in values[candidate]])
            samples = (np.median(b[rng.integers(0, 4, (50000, 4))], axis=1) / np.median(a[rng.integers(0, 4, (50000, 4))], axis=1) - 1) * 100
            lo, hi = map(float, np.quantile(samples, [.025, .975]))
            metrics[metric] = {'effect_percent': float((np.median(b) / np.median(a) - 1) * 100), 'bootstrap95_percent': [lo, hi], 'verdict': verdict(lo, hi, 5 if metric == 'tg_p95_ms' else 3, metric.endswith('_ms'))}
        comparisons[candidate + '_vs_' + base] = metrics
    summary[case] = {'gate': 'ELIGIBLE_BOUNDED_REPLAY', 'process_values': values, 'drift_percent': drift, 'comparisons': comparisons}
out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'gate': 'DESCRIPTIVE_COMPLETE' if len(summary) == 7 else 'DESCRIPTIVE_PARTIAL', 'cells': summary, 'limits': 'All7 inputs reported separately.4 fresh processes per variant; process-level bootstrap50k/seed61. PP+TG decode duration excludes model load,sampling,application TTFT and cold-start qualification. Previously inspected R42 replay corpus,one prompt/family and one fixed continuation. Long correctness failures and M0-M3 acceptance remain open. No default promotion.'}
(q / 'r61-analysis.json').write_text(json.dumps(out, indent=2) + '\n')
for name, cell in summary.items(): print(name, cell['gate'], cell.get('comparisons', {}).get('on_vs_off', {}), flush=True)
