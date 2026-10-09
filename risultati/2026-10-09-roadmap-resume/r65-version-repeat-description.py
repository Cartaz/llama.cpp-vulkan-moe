from pathlib import Path
import datetime, hashlib, json
import numpy as np

q = Path(__file__).resolve().parent
audit = json.loads((q / 'r65-raw-audit.json').read_text())
assert audit['gate'] == 'DESCRIPTIVE_COMPLETE'
cells = {}
for case, cell in audit['cells'].items():
    vectors = (cell['inputs']['PP'] + 511) // 512 + cell['inputs']['TG']
    folder = q / 'r65-model' / case
    records = {}
    for variant in ['B1', 'B4']:
        first = folder / ('raw-' + variant + '-1-logits.bin')
        reference = np.memmap(first, np.uint32, 'r').reshape(3, vectors, 248320)[0]
        for occurrence in range(1, 5):
            name = f'raw-{variant}-{occurrence}'
            path = folder / (name + '-logits.bin')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == cell['processes'][name]['raw_sha256']
            bits = np.memmap(path, np.uint32, 'r').reshape(3, vectors, 248320)
            different = [{'rep': i, 'vector': j} for i, rep in enumerate(bits) for j, value in enumerate(rep) if not np.array_equal(value, reference[j])]
            records[name] = {'variant_first_process_exact': not different, 'different_vectors': len(different), 'first_difference': different[0] if different else None, 'original_B1_gate': cell['processes'][name]['gate']}
    cells[case] = records
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'POSTHOC_DESCRIPTIVE_VERSION_REPEAT', 'selected_after': 'Planning B4 cross-version divergence observed; no acceptance criteria changed', 'cells': cells, 'limits': 'First B4 reference is descriptive only; original chronological B1 gates/failed refs remain authoritative. Exact within-version repeats do not establish semantic equivalence, reliability or eligibility for timing.'}
(q / 'r65-version-repeat-description.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['scope'], flush=True)
