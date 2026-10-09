from pathlib import Path
import json, subprocess, time, datetime, fcntl
q=Path(__file__).resolve().parent
state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'WAITING_POST_R62_AUDIT','protocol':'r63-protocol.json','stages':[],'single_model_process':True}
def save():
 state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (q/'qualification-r63-status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
while not (q/'post-r62-audit-complete.json').exists():time.sleep(1)
assert json.loads((q/'post-r62-audit-complete.json').read_text())['gate']=='POST_RUN_AUDITS_COMPLETE'
lock=(q/'controller-exclusive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
assert json.loads((q/'r63-prelaunch-audit.json').read_text())['gate']=='PASS_PRELAUNCH'
for name,args in [('r63-model-gates',[]),('replay-independent-audit',['r63']),('r63-counters',[]),('r63-timing',[]),('r63-analysis',[])]:
 state.update(status='RUNNING',phase=name);save();start=time.monotonic()
 with (q/(name+'-r63-queue.log')).open('x')as log:
  p=subprocess.Popen(['python3','-u',str(q/(name+'.py'))]+args,cwd=q.parents[1],stdout=log,stderr=subprocess.STDOUT)
  while p.poll()is None:time.sleep(1)
 state['stages'].append({'name':name,'exit':p.returncode,'wall_s':time.monotonic()-start})
 if p.returncode!=0:
  state.update(status='DEPENDENT_GATE_CLOSED',phase='preserved_failure');save();raise SystemExit(1)
state.update(status='R63_BOUNDED_QUALIFICATION_FINISHED',phase='independent_timing_audit_required');save()
