from pathlib import Path
import json, csv, sys, datetime
import numpy as np
q=Path(__file__).resolve().parent;campaign=sys.argv[1];assert campaign in ['r61','r63']
data=json.loads((q/(campaign+'-timing-results.json')).read_text());analysis=json.loads((q/(campaign+'-analysis.json')).read_text())
reps=3 if campaign=='r61'else 1;rng=np.random.default_rng(int(campaign[1:]));errors=[];records={};count=0
keys=['phase','position','n_tokens','logits_hash']
for case,cell in data.items():
 values={};folder=q/(campaign+'-timing')/case
 reference=list(csv.DictReader((folder/'reference.csv').open()));ref=[tuple(x[k]for k in keys)for x in reference]
 for variant in ['resident','old','off','on']:
  values[variant]=[]
  for occurrence in range(1,5):
   name=f'timing-{variant}-{occurrence}'
   try:
    manifest=json.loads((folder/(name+'-result.json')).read_text());env=manifest['environment']
    assert manifest['returncode']==0 and env['MOE_REPLAY_REPS']==str(reps)and env['MOE_REPLAY_WARMUP']==('1'if reps==3 else'0')
    assert not any(k in env for k in ['MOE_REPLAY_LOGITS_OUT','GGML_SCHED_EXPERT_POOL_LOG','GGML_SCHED_PROFILE'])
    assert all(manifest['loaded_libraries'].get(p)==d for p,d in manifest['expected_libraries'].items())
    rows=list(csv.DictReader((folder/(name+'.csv')).open()));assert len(rows)==reps*len(ref)
    for rep in range(reps):assert [tuple(x[k]for k in keys)for x in rows if int(x['rep'])==rep]==ref
    pp=sum(int(x['elapsed_us'])for x in rows if x['phase']=='prefill')/1000
    tg=sum(int(x['elapsed_us'])for x in rows if x['phase']=='decode')/1000
    assert pp>0 and tg>0
    actual={'pp_tps':cell['inputs']['PP']*reps*1000/pp,'tg_tps':cell['inputs']['TG']*reps*1000/tg,'replay_decode_ms':(pp+tg)/reps,'tg_p95_ms':float(np.quantile([int(x['elapsed_us'])/1000 for x in rows if x['phase']=='decode'],.95))}
    values[variant].append(actual);count+=1
   except Exception as e:errors.append({'case':case,'process':name,'error':str(e)})
 records[case]=values
 if any(len(v)!=4 for v in values.values()):continue
 a=analysis['cells'][case];drift={k:(values['resident'][-1][k]/values['resident'][0][k]-1)*100 for k in values['resident'][0]}
 assert all(np.isclose(a['drift_percent'][k],v,rtol=1e-12,atol=1e-12)for k,v in drift.items())
 if any(abs(v)>20 for v in drift.values()):assert a['gate']=='INELIGIBLE';continue
 assert a['gate']=='ELIGIBLE_BOUNDED_REPLAY'
 for base,candidate in [('old','off'),('off','on'),('old','on'),('resident','on')]:
  for metric in values[base][0]:
   aa=np.array([x[metric]for x in values[base]]);bb=np.array([x[metric]for x in values[candidate]])
   samples=(np.median(bb[rng.integers(0,4,(50000,4))],axis=1)/np.median(aa[rng.integers(0,4,(50000,4))],axis=1)-1)*100
   ci=np.quantile(samples,[.025,.975]);point=(np.median(bb)/np.median(aa)-1)*100
   recorded=a['comparisons'][candidate+'_vs_'+base][metric]
   assert np.isclose(point,recorded['effect_percent'],rtol=1e-12,atol=1e-12)
   assert np.allclose(ci,recorded['bootstrap95_percent'],rtol=1e-10,atol=1e-10)
expected=112 if reps==3 else 48
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gate':'PASS_COMPLETE_CSV_CI_AUDIT'if not errors and count==expected else'FAIL_AUDIT','processes':count,'errors':errors,'process_values':records,'checks':'Original CSV hashes/order against original B1 reference; isolated no-raw/no-counter environment; loaded DSOs; PP/TG sums,p95,process medians,drift,50k process bootstrap and preregistered seed independently recalculated. No model loading/application TTFT qualification.'}
(q/(campaign+'-timing-audit.json')).write_text(json.dumps(result,indent=2)+'\n');print(result['gate'],count,flush=True);assert not errors and count==expected
