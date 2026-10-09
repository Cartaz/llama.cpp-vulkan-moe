from pathlib import Path
import json,sys,datetime,csv
import numpy as np
q=Path(__file__).resolve().parent;sys.path.insert(0,str(q))
from r65_common import r,freeze
r.verify_freeze();data=json.loads((q/'r65-model-results.json').read_text());cells={};count=0;total=0
for case,cell in data.items():
    item=cell['inputs'];vectors=(item['PP']+511)//512+item['TG'];folder=q/'r65-model'/case;records={}
    reference=folder/'raw-B1-1-logits.bin'
    ref=np.memmap(reference,np.uint32,'r').reshape(3,vectors,248320)[0]if reference.exists()and reference.stat().st_size==3*vectors*248320*4 else None
    for name in cell['results']:
        record={'gate':'FAIL'};count+=1
        try:
            manifest=json.loads((folder/(name+'-result.json')).read_text());env=manifest['environment']
            assert manifest['returncode']==0 and env['MOE_REPLAY_REPS']=='3'and env['MOE_REPLAY_WARMUP']=='1'
            assert not any(k.startswith('GGML_SCHED_')for k in env)
            assert all(manifest['loaded_libraries'].get(p)==digest for p,digest in manifest['expected_libraries'].items())and len(manifest['expected_libraries'])==6
            raw=folder/(name+'-logits.bin');assert raw.stat().st_size==3*vectors*248320*4
            bits=np.memmap(raw,np.uint32,'r').reshape(3,vectors,248320);floats=bits.view(np.float32)
            finite=all(np.isfinite(v).all()and np.any(v!=0)for rep in floats for v in rep)
            mismatches=[{'rep':i,'vector':j}for i,rep in enumerate(bits)for j,v in enumerate(rep)if ref is not None and not np.array_equal(v,ref[j])]
            repeat=all(np.array_equal(rep,bits[0])for rep in bits)
            geometry=None
            if mismatches:
                first=mismatches[0];av=floats[first['rep'],first['vector']];bv=ref[first['vector']].view(np.float32)
                calls=r.replay_summary.read_calls(folder/(name+'.csv'));call=calls[first['rep']*vectors+first['vector']]
                geometry={k:call[k]for k in ['phase','position','n_tokens']}
                if np.isfinite(av).all()and np.isfinite(bv).all():
                    delta=av.astype(np.float64)-bv.astype(np.float64);geometry.update(max_abs=float(abs(delta).max()),rms=float(np.sqrt(np.mean(delta*delta))),argmax_equal=int(av.argmax())==int(bv.argmax()))
            record.update(gate='PASS'if finite and repeat and ref is not None and not mismatches else'FAIL',finite_nonzero=finite,within_process_exact=repeat,first_B1_available=ref is not None,different_vectors=len(mismatches),first_difference=mismatches[0]if mismatches else None,first_geometry=geometry,raw_sha256=r.sha(raw),vectors=3*vectors)
            total+=3*vectors
        except Exception as error:record['error']=str(error)
        records[name]=record
    cells[case]={'inputs':item,'processes':records,'eligible':len(records)==8 and all(x['gate']=='PASS'for x in records.values()),'timing_allowed':item['timing_allowed']}
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gate':'DESCRIPTIVE_COMPLETE'if count==32 and len(cells)==4 else'AUDIT_PARTIAL','processes':count,'vectors':total,'cells':cells,'limits':'All original attempts classified,including errors/missing first references. Complete vectors and self/fresh parity to chronological B1; upstream differences are not automatically defects. Long diagnostics never eligible for timing; no quality/general qualification.'}
(q/'r65-raw-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result['gate'],count,total,flush=True);assert count==32 and len(cells)==4
