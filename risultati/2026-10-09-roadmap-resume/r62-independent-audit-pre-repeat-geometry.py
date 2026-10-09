from pathlib import Path
import json, hashlib, datetime
import numpy as np
q=Path(__file__).resolve().parent
protocol=json.loads((q/'r62-protocol.json').read_text());freeze=json.loads((q/'r62-freeze.json').read_text())
items=[json.loads(x)for x in (q/'r62-cases.jsonl').read_text().splitlines()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for p,d in freeze['artifacts'].items():assert sha(q/p)==d,p
assert sha(q/'r62-protocol.json')==freeze['protocol_sha256']
assert sha(q/'r62-cases.jsonl')==freeze['cases_sha256']
all_rows={};records={};originals={};vectors=0;valid_cases=0;errors=[]
for name in protocol['order']:
 folder=q/'r62-native'/name
 try:
  manifest=json.loads((folder/(name+'-result.json')).read_text());assert manifest['returncode']==0
  assert all(manifest['loaded_libraries'].get(p)==d==freeze['artifacts'][str(Path(p).relative_to(q))] for p,d in manifest['expected_libraries'].items())
  rows=[json.loads(x)for x in (folder/'responses.jsonl').read_text().splitlines()];assert len(rows)==8
  all_rows[name]={};records[name]={}
  for item,row in zip(items,rows):
   assert row['id']==item['id'] and row['seed']==42 and row['enable_thinking'] and row['temperature']==float(np.float32(.6))
   assert row['prefill_protocol']==('legacy_physical' if '-legacy-'in name else 'application_logical') and row['batch']==2048 and row['ubatch']==512
   p=folder/row['raw_logits_file'];assert p.stat().st_size==row['raw_vectors']*248320*4
   bits=np.memmap(p,np.uint32,'r').reshape(row['raw_vectors'],248320) if row['raw_vectors'] else np.zeros((0,248320),np.uint32)
   bad=[i for i,v in enumerate(bits.view(np.float32))if not np.isfinite(v).all() or not np.any(v!=0)]
   numeric=not bad and row['raw_vectors']>0
   assert numeric==row['numerically_valid'] or row['failure_reason']=='missing_logits'
   if row['numerically_valid']:assert row['raw_vectors']==len(row['generated_token_ids'])
   else:assert row['text']=='' and (folder/(item['id']+'-failure.json')).exists()
   assert (folder/row['raw_text_file']).read_bytes().decode('utf-8','replace')==row['raw_text']
   rec={'numeric_gate':'PASS'if row['numerically_valid']and numeric else 'FAIL','vectors':row['raw_vectors'],'bad_vectors':bad,'raw_sha256':sha(p),'generated_ids':row['generated_token_ids'],'failure_reason':row['failure_reason'],'failure_generation_step':row['failure_generation_step'],'truncated_at64':row['truncated']}
   if item.get('repeat_of'):
    prev=all_rows[name][item['repeat_of']];rec['duplicate_exact']=row['prompt_token_ids']==prev['prompt_token_ids']and row['generated_token_ids']==prev['generated_token_ids']and rec['raw_sha256']==records[name][item['repeat_of']]['raw_sha256']
   key=(name.rsplit('-',1)[0],item['id'])
   if key in originals:
    prev_name,prev=originals[key];rec['fresh_process_exact']=row['prompt_token_ids']==prev['prompt_token_ids']and row['generated_token_ids']==prev['generated_token_ids']and rec['raw_sha256']==records[prev_name][item['id']]['raw_sha256']
   else:originals[key]=(name,row)
   all_rows[name][item['id']]=row;records[name][item['id']]=rec;vectors+=row['raw_vectors'];valid_cases+=rec['numeric_gate']=='PASS'
 except Exception as e:errors.append({'process':name,'error':str(e)})
contrasts={}
for a,b in [('cpu-logical-1','gpu-logical-1'),('cpu-legacy-1','gpu-legacy-1'),('cpu-logical-1','cpu-legacy-1'),('gpu-logical-1','gpu-legacy-1')]:
 cell={}
 for item in items:
  id=item['id']
  if id not in all_rows.get(a,{})or id not in all_rows.get(b,{}):continue
  ar=all_rows[a][id];br=all_rows[b][id]
  ap=q/'r62-native'/a/ar['raw_logits_file'];bp=q/'r62-native'/b/br['raw_logits_file']
  ab=np.memmap(ap,np.uint32,'r').reshape(ar['raw_vectors'],248320) if ar['raw_vectors'] else np.zeros((0,248320),np.uint32)
  bb=np.memmap(bp,np.uint32,'r').reshape(br['raw_vectors'],248320) if br['raw_vectors'] else np.zeros((0,248320),np.uint32)
  diffs=[i for i in range(min(len(ab),len(bb)))if not np.array_equal(ab[i],bb[i])];first=diffs[0]if diffs else None
  same_prompt=ar['prompt_token_ids']==br['prompt_token_ids']
  entry={'prompt_ids_exact':same_prompt,'generated_ids_exact':ar['generated_token_ids']==br['generated_token_ids'],'common_vectors':min(len(ab),len(bb)),'different_common_vectors':len(diffs),'first_different_vector':first,'prior_generated_histories_exact_at_first':same_prompt and ar['generated_token_ids'][:first]==br['generated_token_ids'][:first]if first is not None else None}
  if first is not None:
   av=ab[first].view(np.float32);bv=bb[first].view(np.float32)
   if np.isfinite(av).all()and np.isfinite(bv).all():
    delta=av.astype(np.float64)-bv.astype(np.float64);entry.update(first_max_abs=float(np.max(np.abs(delta))),first_rms=float(np.sqrt(np.mean(delta*delta))),first_argmax_equal=int(av.argmax())==int(bv.argmax()))
  cell[id]=entry
 contrasts[b+'_vs_'+a]=cell
count=sum(len(x)for x in records.values());exact=all(x.get('duplicate_exact',True)and x.get('fresh_process_exact',True)for cell in records.values()for x in cell.values())
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gate':'DESCRIPTIVE_COMPLETE'if not errors and count==64 else 'AUDIT_INCOMPLETE','numerical_repeat_gate':'PASS_BOUNDED_DIAGNOSTIC'if not errors and count==64 and valid_cases==64 and exact else 'FAIL_NUMERICAL_OR_REPEAT','cases':count,'valid_cases':valid_cases,'vectors':vectors,'errors':errors,'processes':records,'contrasts':contrasts,'limits':'Every emitted full sampling vector; first cross-configuration differences indicate matched history only when recorded. Later differing native histories are confounded.64-token diagnostic cap,raw I/O,already warmed drivers; no semantic quality/performance/server reliability qualification.'}
(q/'r62-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result['gate'],result['numerical_repeat_gate'],count,valid_cases,vectors,flush=True)
assert not errors and count==64
