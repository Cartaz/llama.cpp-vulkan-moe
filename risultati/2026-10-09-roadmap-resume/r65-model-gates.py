from pathlib import Path
import json,csv,shutil,collections,sys
import numpy as np
q=Path(__file__).resolve().parent;sys.path.insert(0,str(q))
from r65_common import r,freeze,flags
results={}
for case,item in freeze['inputs'].items():
    r.OUT=q/'r65-model'/case;r.OUT.mkdir(parents=True)
    shutil.copy2(q/item['path'],r.OUT/'tokens.csv')
    vectors=(item['PP']+511)//512+item['TG']
    r.check_raw=lambda p:r.replay_summary.check_logits(p,3*vectors,248320)
    reference=r.OUT/'raw-B1-1-logits.bin';counts=collections.Counter();cell={}
    for variant in ['B1','B4','B4','B1']*2:
        counts[variant]+=1;name='raw-'+variant+'-'+str(counts[variant])
        try:
            assert shutil.disk_usage(q).free>8*1024**3,'Disk reserve below8GiB'
            result=r.run(name,label='R65-'+variant,reps=3,raw=True,flags=flags(item),extra={'MOE_REPLAY_WARMUP':'1'})
            raw=r.OUT/(name+'-logits.bin');bits=np.memmap(raw,np.uint32,'r').reshape(3,vectors,248320)
            assert all(np.array_equal(rep,bits[0])for rep in bits),'Within-process raw divergence'
            assert reference.exists()and reference.stat().st_size==3*vectors*248320*4,'Original first B1 unavailable'
            ref=np.memmap(reference,np.uint32,'r').reshape(3,vectors,248320)[0]
            assert all(np.array_equal(rep,ref)for rep in bits),'Same-GPU first B1 raw divergence'
            if name=='raw-B1-1':
                calls=r.replay_summary.read_calls(r.OUT/(name+'.csv'))
                with (r.OUT/'reference.csv').open('x',newline='')as f:
                    writer=csv.DictWriter(f,fieldnames=r.replay_summary.CALL_FIELDS);writer.writeheader();writer.writerows(x for x in calls if x['rep']==0)
            else:assert result['metrics']['per_call_matches_reference'],'Fingerprint mismatch'
            result.update(gate='PASS',raw_vectors=3*vectors,raw_sha256=r.sha(raw))
        except Exception as error:
            p=r.OUT/(name+'-result.json');result=json.loads(p.read_text())if p.exists()else{}
            result.update(gate='FAIL',error=str(error))
        cell[name]=result
        results[case]={'inputs':item,'results':cell,'eligible':len(cell)==8 and all(x['gate']=='PASS'for x in cell.values())}
        (q/'r65-model-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(case,name,result['gate'],result.get('error',''),flush=True)
print('All32 attempted raw processes retained; eligibility remains per-cell, no long timing',flush=True)
