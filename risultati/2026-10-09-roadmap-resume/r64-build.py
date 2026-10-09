from pathlib import Path
import json, hashlib, shutil, subprocess, datetime, os
q = Path(__file__).resolve().parent
root = q.parents[1]
registered = json.loads((q/'r64-source-freeze.json').read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in registered['sources'].items():
    assert sha(q/name)==digest,name
for name,digest in registered['headers'].items():
    assert sha(root/name)==digest,name
parent = json.loads((q/'r62-freeze.json').read_text())
for name,digest in parent['artifacts'].items():
    assert sha(q/name)==digest,name
out = q/'frozen/R64-BASE'
out.mkdir()
for p in (q/'frozen/R62-BASE').glob('*.so*'):
    target=out/p.name
    if p.is_symlink(): os.symlink(os.readlink(p),target)
    else: shutil.copy2(p,target)
command = json.loads((q/'r62-compile.json').read_text())['argv']
command=[x.replace('r62-native-prefill.cpp','r64-context-lifecycle.cpp').replace('frozen/R62-BASE','frozen/R64-BASE')for x in command]
result=subprocess.run(command,cwd=root,capture_output=True,text=True)
(q/'r64-compile.stdout').write_text(result.stdout)
(q/'r64-compile.stderr').write_text(result.stderr)
record={'argv':command,'returncode':result.returncode,'source_sha256':sha(q/'r64-context-lifecycle.cpp'),'compiler':subprocess.check_output(['g++','--version'],text=True),'headers':registered['headers'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(q/'r64-compile.json').write_text(json.dumps(record,indent=2)+'\n')
assert result.returncode==0,result.stderr
freeze=dict(parent)
freeze['protocol_sha256']=sha(q/'r64-protocol.json')
freeze['helper_source_sha256']=sha(q/'r64-context-lifecycle.cpp')
freeze['variants']={'R64-BASE':'B1 unmodified7fe450e DSOs; archived uniform context-lifecycle helper only'}
freeze['artifacts']={name.replace('frozen/R62-BASE','frozen/R64-BASE'):digest for name,digest in parent['artifacts'].items()if '/lib'in name}
freeze['artifacts']['frozen/R64-BASE/corpus']=sha(out/'corpus')
freeze['r64_sources']=registered['sources']
freeze['r64_headers']=registered['headers']
freeze['source_freeze_sha256']=sha(q/'r64-source-freeze.json')
freeze['utc']=record['utc']
(q/'r64-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print('R64 helper compiled; original B1 DSOs unchanged',flush=True)
