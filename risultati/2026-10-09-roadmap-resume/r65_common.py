from pathlib import Path
import json,sys
q=Path(__file__).resolve().parent
sys.path.insert(0,str(q))
import guarded_runner as r
freeze=json.loads((q/'r65-freeze.json').read_text())
r.CAMPAIGN=q;r.FROZEN=q/'frozen';r.FREEZE=freeze
r.workload=lambda:r.replay_summary.read_tokens(r.OUT/'tokens.csv')
r.REFERENCE='reference.csv'
def verify():
    for name,digest in freeze['artifacts'].items():assert r.sha(q/name)==digest,name
    assert r.sha(q/'r65-protocol.json')==freeze['protocol_sha256']
    assert r.sha(q/'r65-source-freeze.json')==freeze['source_freeze_sha256']
    for name,digest in freeze['sources'].items():assert r.sha(q/name)==digest,name
    for item in freeze['inputs'].values():assert r.sha(q/item['path'])==item['sha256']
r.verify_freeze=verify
def flags(item):
    f=list(r.FLAGS)+['--load-mode','mmap']
    f[f.index('-c')+1]=str(item['context']);f[f.index('-b')+1]=str(item['batch'])
    f[f.index('-ncmoe'):f.index('-ncmoe')]=['-ot','^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0']
    return f
