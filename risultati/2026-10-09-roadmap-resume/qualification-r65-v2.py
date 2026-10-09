from pathlib import Path
import json,subprocess,time,datetime,fcntl
q=Path(__file__).resolve().parent
state={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'WAITING_POST_R64_AUDIT','protocol':'r65-protocol.json','stages':[],'single_model_process':True,'build_jobs':2}
def save():
    state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (q/'qualification-r65-v2-status.json').write_text(json.dumps(state,indent=2)+'\n')
save()
while not(q/'post-r64-audit-complete.json').exists():time.sleep(1)
assert json.loads((q/'post-r64-audit-complete.json').read_text())['gate']=='POST_R64_AUDIT_COMPLETE'
lock=(q/'controller-exclusive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
assert json.loads((q/'r65-prelaunch-audit.json').read_text())['gate']=='PASS_SOURCE_PROTOCOL_PRELAUNCH'
for stage in ['r65-build-v2','r65-model-gates','r65-raw-audit','r65-timing','r65-analysis','r65-timing-audit']:
    state.update(status='RUNNING',phase=stage);save()
    with(q/(stage+'-queue.log')).open('x')as log:
        start=time.monotonic();p=subprocess.Popen(['python3','-u',str(q/(stage+'.py'))],cwd=q.parents[1],stdout=log,stderr=subprocess.STDOUT)
        while p.poll()is None:time.sleep(1)
    state['stages'].append({'name':stage,'exit':p.returncode,'wall_s':time.monotonic()-start})
    if p.returncode:
        state.update(status='DEPENDENT_GATE_CLOSED',phase='preserved_failure');save();raise SystemExit(p.returncode)
state.update(status='R65_BOUNDED_QUALIFICATION_FINISHED',phase='root_review_and_publish_required');save()
