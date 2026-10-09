from pathlib import Path
import json,sys,datetime
import numpy as np
q=Path(__file__).resolve().parent;sys.path.insert(0,str(q))
import guarded_runner as r
data=json.loads((q/'r65-timing-results.json').read_text());rng=np.random.default_rng(65);cells={}
def verdict(lo,hi,margin,latency):
    if lo>=-margin and hi<=margin:return 'NESSUN_CAMBIAMENTO'
    if (latency and hi<-margin)or(not latency and lo>margin):return 'MIGLIORAMENTO'
    if (latency and lo>margin)or(not latency and hi<-margin):return 'REGRESSIONE'
    return 'INCONCLUDENTE'
for case,cell in data.items():
    if not cell['eligible']:
        cells[case]={'gate':'INELIGIBLE','reason':cell.get('reason','Incomplete/failed timing gate')};continue
    values={}
    for variant in ['B1','B4']:
        records=[]
        for occurrence in range(1,5):
            name='timing-'+variant+'-'+str(occurrence)
            calls=r.replay_summary.read_calls(q/'r65-timing'/case/(name+'.csv'))
            pp=sum(x['elapsed_us']for x in calls if x['phase']=='prefill')/1000
            tg=sum(x['elapsed_us']for x in calls if x['phase']=='decode')/1000
            records.append({'pp_tps':cell['inputs']['PP']*3*1000/pp,'tg_tps':cell['inputs']['TG']*3*1000/tg,'replay_decode_ms':(pp+tg)/3,'tg_p95_ms':r.replay_summary.percentile([x['elapsed_us']/1000 for x in calls if x['phase']=='decode'],.95)})
        values[variant]=records
    drift={k:(values['B1'][-1][k]/values['B1'][0][k]-1)*100 for k in values['B1'][0]}
    if any(abs(v)>20 for v in drift.values()):
        cells[case]={'gate':'INELIGIBLE','reason':'B1first-last drift exceeds20%','drift_percent':drift,'process_values':values};continue
    effects={}
    for metric in values['B1'][0]:
        a=np.array([x[metric]for x in values['B1']]);b=np.array([x[metric]for x in values['B4']])
        samples=(np.median(b[rng.integers(0,4,(50000,4))],axis=1)/np.median(a[rng.integers(0,4,(50000,4))],axis=1)-1)*100
        lo,hi=map(float,np.quantile(samples,[.025,.975]))
        effects[metric]={'effect_percent':float((np.median(b)/np.median(a)-1)*100),'bootstrap95_percent':[lo,hi],'verdict':verdict(lo,hi,5 if metric=='tg_p95_ms'else 3,metric.endswith('_ms'))}
    cells[case]={'gate':'ELIGIBLE_BOUNDED_REPLAY','process_values':values,'drift_percent':drift,'B4_vs_B1':effects}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gate':'DESCRIPTIVE_COMPLETE'if len(cells)==2 else'DESCRIPTIVE_PARTIAL','cells':cells,'limits':'Two previously inspected short fixed replays;4freshprocesses/variant,3warmmeasuredreps,50kprocess bootstrap seed65. No long timing even if raw passes. Full unmodified upstream79e versus unmodifiedB1; bundled changes, no PR attribution. No native quality,application TTFT,coldstart,concurrency,default/integration or M0-M3 qualification.'}
(q/'r65-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(result['gate'],[(k,v['gate'])for k,v in cells.items()],flush=True)
