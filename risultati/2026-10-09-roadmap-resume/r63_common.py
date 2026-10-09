from pathlib import Path
import json,sys
q=Path(__file__).resolve().parent
sys.path.insert(0,str(q))
import guarded_runner as r
freeze=json.loads((q/'r61-model-freeze.json').read_text())
registered=json.loads((q/'r63-freeze.json').read_text())
freeze['inputs']={n:freeze['inputs'][n]for n in registered['cases']}
r.CAMPAIGN=q;r.FROZEN=q/'frozen';r.FREEZE=freeze
r.workload=lambda:r.replay_summary.read_tokens(r.OUT/'tokens.csv')
r.REFERENCE='reference.csv'
def verify():
 for name,digest in freeze['artifacts'].items():assert r.sha(q/name)==digest,name
 assert r.sha(q/'r61-protocol.json')==freeze['protocol_sha256']
 assert r.sha(q/'r63-protocol.json')==registered['protocol_sha256']
 for name,digest in registered['sources'].items():assert r.sha(q/name)==digest,name
 for item in freeze['inputs'].values():assert r.sha(q/item['path'])==item['sha256']
r.verify_freeze=verify
