"""Read full raw data and original CSV/manifests without trusting recorded PASS flags."""
from pathlib import Path
import json, sys, hashlib, csv, datetime
import numpy as np
q=Path(__file__).resolve().parent
campaign=sys.argv[1]
assert campaign in ['r61','r63']
cold=campaign=='r63'
freeze=json.loads((q/'r61-model-freeze.json').read_text())
registered=json.loads((q/'r63-freeze.json').read_text()) if cold else None
inputs={k:v for k,v in freeze['inputs'].items() if not cold or k in registered['cases']}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for p,digest in freeze['artifacts'].items():assert sha(q/p)==digest,p
for item in inputs.values():assert sha(q/item['path'])==item['sha256']
if cold:
 assert sha(q/'r63-protocol.json')==registered['protocol_sha256']
 for p,digest in registered['sources'].items():assert sha(q/p)==digest,p
out={}; total=0; good=True
for case,item in inputs.items():
 folder=q/(campaign+'-model')/case; vectors=(item['PP']+511)//512+item['TG']; reps=1 if cold else 3
 refpath=folder/('cold-gate-resident-1-logits.bin' if cold else 'gate-resident-logits.bin')
 ref=np.memmap(refpath,np.uint32,'r').reshape(reps,vectors,248320)[0]
 warm=np.memmap(q/'r61-model'/case/'gate-resident-logits.bin',np.uint32,'r').reshape(3,vectors,248320)[0]
 out[case]={}
 for variant in ['resident','old','off','on']:
  for occurrence in range(1,5) if cold else [1]:
   name=f'cold-gate-{variant}-{occurrence}' if cold else 'gate-'+variant
   record={'gate':'FAIL'}
   try:
    manifest=json.loads((folder/(name+'-result.json')).read_text());env=manifest['environment']
    assert manifest['returncode']==0 and manifest['reps']==reps
    assert env['MOE_REPLAY_WARMUP']==('0' if cold else '1') and env['MOE_REPLAY_REPS']==str(reps)
    assert (env.get('GGML_SCHED_EXPERT_POOL_DECODE_ONLY')=='1')==(variant=='on')
    assert sha(Path(env['MOE_REPLAY_IN']))==item['sha256']
    expected=manifest['expected_libraries'];assert len(expected)==6
    assert all(manifest['loaded_libraries'].get(p)==digest==freeze['artifacts'][str(Path(p).relative_to(q))] for p,digest in expected.items())
    raw=folder/(name+'-logits.bin');assert raw.stat().st_size==reps*vectors*248320*4
    bits=np.memmap(raw,np.uint32,'r').reshape(reps,vectors,248320);floats=bits.view(np.float32)
    finite=all(np.isfinite(v).all() and np.any(v!=0) for rep in floats for v in rep)
    mismatches=[{'rep':i,'vector':j} for i,rep in enumerate(bits) for j,v in enumerate(rep) if not np.array_equal(v,ref[j])]
    warm_equal=all(np.array_equal(rep,warm)for rep in bits)
    calls=list(csv.DictReader((folder/(name+'.csv')).open()));assert len(calls)==reps*vectors
    admitted=0;bypasses=0
    if variant!='resident':
     rows=list(csv.DictReader((folder/(name+'-pool.csv')).open()));blocks=1 if cold else 4
     assert len(rows)==blocks*(1+item['TG']) and int(rows[0]['hits'])==0
     admitted=sum(x['reason']=='ready' and int(x['n_tokens'])==1 and int(x['projections'])==3 for x in rows)
     assert admitted>0
     if variant=='on':
      skipped=[x for x in rows if x['reason']=='batched_bypass'];bypasses=len(skipped)
      assert bypasses==blocks and all(all(int(x[k])==0 for k in ['projections','hits','misses','evictions','upload_bytes'])for x in skipped)
      assert not any(x['reason']=='ready' and int(x['n_tokens'])>1 for x in rows)
    record.update(gate='PASS' if finite and not mismatches and warm_equal else 'FAIL',vectors=reps*vectors,finite_nonzero=finite,raw_sha256=sha(raw),mismatches=mismatches,warm_B1_exact=warm_equal,admitted_TG=admitted,batched_bypasses=bypasses)
    total+=reps*vectors
   except Exception as error:record['error']=str(error)
   good &= record['gate']=='PASS';out[case][name]=record
expected_count=48 if cold else 28
count=sum(len(v)for v in out.values())
gate=('PASS_ALL48_FULL_RAW' if cold else 'PASS_ALL28_FULL_RAW') if good and count==expected_count else 'FAIL_FULL_RAW_AUDIT'
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gate':gate,'processes':count,'vectors':total,'cases':out,'limits':'Complete float32 vectors; fixed teacher-forced histories, same-GPU B1. No semantic/native generation, long-context or general correctness qualification.'}
(q/(campaign+'-raw-audit.json')).write_text(json.dumps(result,indent=2)+'\n')
print(gate,count,total,flush=True)
assert good and count==expected_count
