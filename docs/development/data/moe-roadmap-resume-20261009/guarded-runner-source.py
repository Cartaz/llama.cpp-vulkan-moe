import csv, hashlib, json, os, re, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import importlib.util

spec=importlib.util.spec_from_file_location("replay_summary",Path('/home/casa/Programmi/Ottimizzazione_llama_cpp_moe/llama-moe-ab/risultati/2026-10-08-quantized-work-replay/replay-summary.py'))
replay_summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay_summary)

ROOT=Path('/home/casa/Programmi/Ottimizzazione_llama_cpp_moe/llama-moe-ab')
CAMPAIGN=ROOT/'risultati/2026-10-08-quantized-work-replay'
OUT=Path(__file__).resolve().parent/'short'
FROZEN=CAMPAIGN/'frozen'
DEVICE=Path('/sys/bus/pci/devices/0000:09:00.0')
FREEZE=json.loads((CAMPAIGN/'freeze.json').read_text())
MODEL=FREEZE['model']['path']
REFERENCE='old-cpu.csv'
FLAGS=['-m',MODEL,'-ngl','99','-ncmoe','18','-t','8','-tb','8','-c','1024','-b','512','-ub','512','-fa','on','-ctk','q8_0','-ctv','q8_0','--fit','off','--verbosity','4']


def read(p):
    try: return Path(p).read_text().strip()
    except OSError: return ''


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def meminfo():
    return {k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in read('/proc/meminfo').splitlines())}


def verify_freeze():
    for section in ['artifacts','source','input_files']:
        for name,digest in FREEZE[section].items(): assert sha(CAMPAIGN/name)==digest,name
    assert Path(MODEL).stat().st_size==FREEZE['model']['bytes']
    assert sha(CAMPAIGN/'protocol.json')==FREEZE['protocol_sha256']
    case=FREEZE['cases'][0]
    for name,key in [('prompt.txt','prompt_sha256'),('prompt-ids.json','prompt_ids_sha256')]: assert sha(OUT/name)==case[key]
    wf=json.loads((OUT/'workload-freeze.json').read_text())
    assert sha(OUT/'tokens.csv')==wf['workload_sha256']


def gpu_clients(pid):
    clients={}
    for p in Path('/proc',str(pid),'fdinfo').glob('*'):
        data=read(p)
        if 'drm-driver:\tamdgpu' not in data: continue
        fields={k:v.strip() for k,v in (line.split(':',1) for line in data.splitlines() if ':' in line)}
        clients[fields.get('drm-client-id',p.name)]={k:v for k,v in fields.items() if k.startswith('drm-')}
    return list(clients.values())


def workload():
    prompt,decode=replay_summary.read_tokens(OUT/'tokens.csv')
    case=next(c for c in FREEZE['cases'] if c['name']==OUT.name)
    assert (prompt,decode)==(case['prompt_tokens'],case['decode_tokens'])
    return prompt,decode


def validate_trace():
    prompt,decode=workload()
    case=next(c for c in FREEZE['cases'] if c['name']==OUT.name)
    ids=json.loads((OUT/'prompt-ids.json').read_text())
    actual=[int(r['id']) for r in csv.DictReader((OUT/'tokens.csv').open()) if r['phase']=='prefill']
    assert actual==ids and len(actual)==case['prompt_tokens'], 'Tokenizer/file parsing mismatch'
    return {'prefill_tokens':prompt,'decode_tokens':decode,'prompt_ids_match':True,'routing':'Not recorded: baseline no-callback helper'}


def rows_metrics(path):
    summary=replay_summary.summarize(path,OUT/'tokens.csv',512)
    reference=OUT/REFERENCE
    if reference.exists():
        try:
            replay_summary.summarize(path,OUT/'tokens.csv',512,reference)
            summary['per_call_matches_reference']=True
        except ValueError as error:
            summary['per_call_matches_reference']=False
            summary['reference_error']=str(error)
    return summary


def check_raw(path):
    prompt,decode=workload()
    return replay_summary.check_logits(path,(prompt+511)//512+decode,248320)


def run(name,label='NEW',profile=False,reps=3,raw=False,extra=None,expect=0,flags=None,mode='replay'):
    verify_freeze()
    assert not (OUT/(name+'.log')).exists(), 'Preserve completed/partial run'
    assert meminfo()['MemAvailable']>=6144*1024**2, 'Insufficient RAM before load'
    probe=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True,check=True)
    assert 'AMD Radeon RX 6800 (RADV NAVI21)' in probe.stdout, 'RX 6800 unavailable'
    assert int(read(DEVICE/'mem_info_vram_used'))<3*1024**3, 'Concurrent VRAM workload'
    env={k:os.environ[k] for k in ('PATH','HOME','USER','XDG_RUNTIME_DIR','DISPLAY','WAYLAND_DISPLAY') if k in os.environ}
    env.update(LC_ALL='C',LD_LIBRARY_PATH=str(FROZEN/label),MOE_REPLAY_IN=str(OUT/'tokens.csv'),MOE_REPLAY_REPS=str(reps))
    if profile:
        env.update(GGML_SCHED_PROFILE=str(OUT/(name+'-scheduler.csv')),MOE_REPLAY_PROFILE=str(OUT/(name+'-phases.csv')))
    if raw: env['MOE_REPLAY_LOGITS_OUT']=str(OUT/(name+'-logits.bin'))
    env.update(extra or {})
    if mode=='trace':
        env.pop('MOE_REPLAY_IN');env.pop('MOE_REPLAY_REPS')
        env.pop('MOE_REPLAY_LOGITS_OUT',None)
        env.update(MOE_TRACE_OUT=str(OUT/'routing.csv'),MOE_TRACE_TOKENS_OUT=str(OUT/'tokens.csv'),MOE_TRACE_LOGITS_OUT=str(OUT/(name+'-logits.bin')))
    if mode in ['corpus','check','state']:
        for key in ['MOE_REPLAY_IN','MOE_REPLAY_REPS','MOE_REPLAY_LOGITS_OUT']: env.pop(key,None)
    unit='moe-r44-'+OUT.name+'-'+name.lower()
    argv=[str(FROZEN/label/('state-check' if mode=='state' else 'scheduler-check' if mode=='check' else 'corpus' if mode=='corpus' else 'trace' if mode=='trace' else 'replay')),*(flags or FLAGS)]
    launch=['systemd-run','--user','--wait','--pipe','--collect','--unit='+unit,'-p','MemoryMax=24G','-p','MemorySwapMax=2G','-p','OOMPolicy=stop','-p','WorkingDirectory='+str(ROOT),'/usr/bin/env','-i',*[k+'='+v for k,v in env.items()],*argv]
    manifest={'argv':argv,'launch':launch,'environment':env,'started_utc':datetime.now(timezone.utc).isoformat(),'source_commit':FREEZE['variants'][label],'helper_commit':FREEZE['candidate_commit'],'helper_source_sha256':FREEZE['helper_source_sha256'],'label':label,'profile':profile,'reps':reps,'clocks':read(DEVICE/'pp_od_clk_voltage'),'power_cap':{p.name:read(p) for p in DEVICE.glob('hwmon/hwmon*/power*_cap')},'gpu_profile':read(DEVICE/'pp_power_profile_mode')}
    (OUT/(name+'-planned.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    sensors={key:DEVICE/key for key in ['gpu_busy_percent','mem_info_vram_used','mem_info_gtt_used','pp_dpm_sclk','pp_dpm_mclk']}
    sensors.update({p.name:p for p in DEVICE.glob('hwmon/hwmon*/*') if p.name.startswith(('temp','power','freq')) and p.name.endswith(('_input','_average'))})
    start=time.monotonic(); pid=0; cg=None; maps=''; stop_reason=None
    with (OUT/(name+'.csv')).open('w') as stdout,(OUT/(name+'.log')).open('w') as stderr,(OUT/(name+'-telemetry.csv')).open('w') as telemetry,(OUT/(name+'-drm.jsonl')).open('w') as drm:
        writer=csv.DictWriter(telemetry,fieldnames=['elapsed_s','pid','rss_kib','hwm_kib','mem_available','swap_free','cgroup_memory_current','cgroup_memory_peak','cgroup_swap_current','pss_kib','private_clean_kib','private_dirty_kib','cpus_allowed_list',*sensors])
        writer.writeheader()
        p=subprocess.Popen(launch,stdout=stdout,stderr=stderr,cwd=ROOT)
        try:
            while p.poll() is None:
                if not pid:
                    info=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ControlGroup'],capture_output=True,text=True).stdout
                    props=dict(line.split('=',1) for line in info.splitlines() if '=' in line)
                    pid=int(props.get('MainPID','0'))
                    cg=Path('/sys/fs/cgroup'+props['ControlGroup']) if props.get('ControlGroup') else None
                memory=meminfo()
                status={k:v.strip() for k,v in (line.split(':',1) for line in read(f'/proc/{pid}/status').splitlines() if ':' in line)} if pid else {}
                row={'elapsed_s':round(time.monotonic()-start,3),'pid':pid,'rss_kib':status.get('VmRSS','').split(' ')[0],'hwm_kib':status.get('VmHWM','').split(' ')[0],'mem_available':memory['MemAvailable'],'swap_free':memory['SwapFree']}
                if pid:
                    rollup={k:v.strip() for k,v in (line.split(':',1) for line in read(f'/proc/{pid}/smaps_rollup').splitlines() if ':' in line)}
                    row.update(pss_kib=rollup.get('Pss','').split(' ')[0],private_clean_kib=rollup.get('Private_Clean','').split(' ')[0],private_dirty_kib=rollup.get('Private_Dirty','').split(' ')[0],cpus_allowed_list=status.get('Cpus_allowed_list',''))
                    if 'cpu_affinity' not in manifest and status.get('Cpus_allowed_list'):
                        manifest['cpu_affinity']={'allowed_list':status['Cpus_allowed_list'],'observed_elapsed_s':time.monotonic()-start,'scope':'one live observation; inherited affinity, not pinned'}
                if cg:
                    row.update(cgroup_memory_current=read(cg/'memory.current'),cgroup_memory_peak=read(cg/'memory.peak'),cgroup_swap_current=read(cg/'memory.swap.current'))
                row.update({key:read(path) for key,path in sensors.items()});writer.writerow(row);telemetry.flush()
                if pid:
                    drm.write(json.dumps({'elapsed_s':time.monotonic()-start,'clients':gpu_clients(pid)})+'\n');drm.flush()
                    current=read(f'/proc/{pid}/maps')
                    if 'libggml-base.so' in current: maps=current
                if memory['MemAvailable']<6144*1024**2: stop_reason='MemAvailable below 6 GiB'
                if time.monotonic()-start>600: stop_reason='600s timeout'
                if pid and (mode!='check' or '--vulkan' in (flags or [])) and time.monotonic()-start>3 and 'RADV NAVI21' not in read(OUT/(name+'.log')): stop_reason='Vulkan device was not confirmed before load'
                if stop_reason:
                    subprocess.run(['systemctl','--user','stop',unit],capture_output=True)
                    break
                time.sleep(.5)
            p.wait(timeout=30)
        finally:
            if p.poll() is None:
                subprocess.run(['systemctl','--user','stop',unit],capture_output=True);p.wait(timeout=30)
    if label=='ASAN':
        log=read(OUT/(name+'.log'))
        assert not any(s in log for s in ['ERROR: AddressSanitizer','ERROR: LeakSanitizer','runtime error:']), 'Preserve sanitizer failure'
    manifest.update(returncode=p.returncode,wall_s=time.monotonic()-start,pid=pid,stop_reason=stop_reason)
    (OUT/(name+'-maps.txt')).write_text(maps+'\n')
    paths={line.split(maxsplit=5)[5] for line in maps.splitlines() if len(line.split(maxsplit=5))==6 and line.split(maxsplit=5)[5].startswith('/') and '.so' in line.split(maxsplit=5)[5]}
    assert all(not path.endswith(' (deleted)') for path in paths)
    manifest['loaded_libraries']={path:sha(path) for path in sorted(paths)}
    expected_libs={str(CAMPAIGN/name):digest for name,digest in FREEZE['artifacts'].items() if name.startswith('frozen/'+label+'/lib') and not name.endswith('/libllama-bench-impl.so') and (mode!='check' or 'libllama' not in name)}
    manifest['expected_libraries']=expected_libs
    (OUT/(name+'-result.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    assert expected_libs and all(manifest['loaded_libraries'].get(path)==digest for path,digest in expected_libs.items()), 'Wrong loaded engine libraries'
    manifest['init_lines']=[line for line in read(OUT/(name+'.log')).splitlines() if re.search(r'offloaded .*layers|buffer size|Flash Attention|replay:|n_ctx =|n_batch =|n_ubatch =',line)]
    (OUT/(name+'-result.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    if p.returncode==0:
        if mode=='trace': manifest['trace_validation']=validate_trace()
        if raw: manifest['raw']=check_raw(OUT/(name+'-logits.bin'))
        if mode=='replay': manifest['metrics']=rows_metrics(OUT/(name+'.csv'))
        if profile:
            subprocess.run(['python3',str(ROOT/'examples/moe-trace/profile-summary.py'),str(OUT/(name+'-scheduler.csv')),'--phases',str(OUT/(name+'-phases.csv')),'--json',str(OUT/(name+'-summary.json'))],check=True)
    (OUT/(name+'-result.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    verify_freeze()
    print(name,'exit',p.returncode,manifest.get('metrics'),flush=True)
    if p.returncode!=expect: raise RuntimeError(name+' failed; inspect preserved artifacts')
    return manifest
