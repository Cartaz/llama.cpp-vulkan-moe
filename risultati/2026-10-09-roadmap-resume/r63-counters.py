from pathlib import Path
import csv, json, datetime
q = Path(__file__).resolve().parent
gates = json.loads((q / 'r63-model-results.json').read_text())
summary = {}
for case, cell in gates.items():
    out = {}
    for variant in ['old', 'off', 'on']:
      for occurrence in range(1,5):
          result = cell['results']['cold-gate-' + variant + '-' + str(occurrence)]
          if result['gate'] != 'PASS':
              out[variant+'-'+str(occurrence)] = {'gate': 'INELIGIBLE'}
              continue
          rows = list(csv.DictReader((q / 'r63-model' / case / ('cold-gate-' + variant + '-' + str(occurrence) + '-pool.csv')).open()))
          assert len(rows) == 1 + cell['inputs']['TG']
          records = []
          for block in range(1):
              batch = rows[block * (1 + cell['inputs']['TG']):(block + 1) * (1 + cell['inputs']['TG'])]
              phases = {}
              for phase, selected in [('PP', batch[:1]), ('TG', batch[1:])]:
                  totals = {k: sum(int(x[k]) for x in selected) for k in ['hits', 'misses', 'evictions', 'upload_bytes']}
                  totals.update(calls=len(selected), admitted_calls=sum(x['reason'] == 'ready' and int(x['projections']) == 3 for x in selected), full_request_hits=sum(x['reason'] == 'ready' and int(x['projections']) == 3 and int(x['misses']) == 0 for x in selected), reasons={reason: sum(x['reason'] == reason for x in selected) for reason in sorted({x['reason'] for x in selected})})
                  denom = totals['hits'] + totals['misses']
                  totals['expert_hit_fraction'] = totals['hits'] / denom if denom else None
                  phases[phase] = totals
              records.append({'block': 'first_replay_warmup_OFF', 'phases': phases})
          out[variant+'-'+str(occurrence)] = {'gate': 'DESCRIPTIVE_COUNTERS', 'blocks': records, 'max_pool_allocated_bytes': max(int(x['pool_bytes']) for x in rows), 'limits': 'Pool-upload bytes only; zero does not imply zero total PCIe transfers. Full-request hits cover selected layer17,not all40 layers. First model/context/expert-cache block is not a controlled cold shader-cache timing.'}
    summary[case] = out
(q / 'r63-counters.json').write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cases': summary}, indent=2) + '\n')
print('R63 cold/warm PP/TG counters recorded separately for all3cases/4freshprocesses', flush=True)
