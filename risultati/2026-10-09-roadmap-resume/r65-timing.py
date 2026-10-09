from pathlib import Path
import json,sys,shutil,collections
q=Path(__file__).resolve().parent;sys.path.insert(0,str(q))
from r65_common import r,freeze,flags
audit=json.loads((q/'r65-raw-audit.json').read_text());assert audit['gate']=='DESCRIPTIVE_COMPLETE'
results={}
for case,item in freeze['inputs'].items():
    if not item['timing_allowed']:continue
    if not audit['cells'][case]['eligible']:
        results[case]={'inputs':item,'results':{},'eligible':False,'reason':'Original full raw gate failed; preserve every attempt'}
        (q/'r65-timing-results.json').write_text(json.dumps(results,indent=2)+'\n');continue
    r.OUT=q/'r65-timing'/case;r.OUT.mkdir(parents=True)
    shutil.copy2(q/item['path'],r.OUT/'tokens.csv');shutil.copy2(q/'r65-model'/case/'reference.csv',r.OUT/'reference.csv')
    counts=collections.Counter();cell={}
    for variant in ['B1','B4','B4','B1']*2:
        counts[variant]+=1;name='timing-'+variant+'-'+str(counts[variant])
        try:
            assert shutil.disk_usage(q).free>8*1024**3,'Disk reserve below8GiB'
            result=r.run(name,label='R65-'+variant,reps=3,flags=flags(item),extra={'MOE_REPLAY_WARMUP':'1'})
            assert result['metrics']['per_call_matches_reference'],'Timing fingerprint mismatch'
            result['gate']='PASS'
        except Exception as error:
            p=r.OUT/(name+'-result.json');result=json.loads(p.read_text())if p.exists()else{}
            result.update(gate='FAIL',error=str(error))
        cell[name]=result
        results[case]={'inputs':item,'results':cell,'eligible':len(cell)==8 and all(x['gate']=='PASS'for x in cell.values())}
        (q/'r65-timing-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(case,name,result['gate'],result.get('error',''),flush=True)
print('Both short cells reported, no long timing or omitted failures',flush=True)
