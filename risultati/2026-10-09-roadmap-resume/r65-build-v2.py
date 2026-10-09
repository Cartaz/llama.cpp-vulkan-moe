from pathlib import Path
import json, hashlib, shutil, subprocess, os, datetime
q=Path(__file__).resolve().parent
root=q.parents[1]
registered=json.loads((q/'r65-source-freeze.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
for name,digest in registered['sources'].items():assert sha(q/name)==digest,name
protocol=json.loads((q/'r65-protocol.json').read_text())
parent=json.loads((q/'r61-model-freeze.json').read_text())
for name,digest in parent['artifacts'].items():assert sha(q/name)==digest,name
model=Path(parent['model']['path']);assert model.stat().st_size==parent['model']['bytes']and sha(model)==parent['model']['sha256'],'Model identity changed'
assert shutil.disk_usage(q).free>12*1024**3,'Build disk reserve below12GiB'
src=q/'r65-upstream-src-v2';build=q/'r65-upstream-build-v2'
assert not src.exists()and not build.exists(),'Preserve partial build/source'
commands=[]
def run(name,argv,cwd=root):
    commands.append({'name':name,'argv':argv,'cwd':str(cwd)})
    with (q/('r65-build-v2-'+name+'.log')).open('x')as log:
        p=subprocess.run(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"CMAKE_BUILD_PARALLEL_LEVEL":"2"})
    (q/'r65-build-v2-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    assert p.returncode==0,name+' failed; preserve log'
head='79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6'
run('clone',['git','clone','--no-hardlinks','--no-checkout',str(root),str(src)])
run('fetch',['git','fetch','--no-tags','https://github.com/ggml-org/llama.cpp.git',head],src)
run('checkout',['git','checkout','--detach',head],src)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==head
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src).strip()
run('configure',['cmake','-S',str(src),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=/usr/bin/gcc','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_C_FLAGS=','-DCMAKE_CXX_FLAGS=','-DBUILD_SHARED_LIBS=ON','-DGGML_NATIVE=ON','-DGGML_OPENMP=ON','-DGGML_BACKEND_DL=OFF','-DGGML_VULKAN=ON','-DGGML_CUDA=OFF','-DLLAMA_BUILD_COMMON=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_MTMD=OFF'])
run('compile',['cmake','--build',str(build),'--target','llama-common','ggml-cpu','ggml-vulkan','-j','2'])
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src).strip(),'Upstream source changed'
artifacts={}
for label,origin in [('R65-B1',q/'frozen/R61-BASE'),('R65-B4',build/'bin')]:
    target=q/'frozen'/label;target.mkdir()
    for p in origin.glob('*.so*'):
        if p.is_symlink():os.symlink(os.readlink(p),target/p.name)
        else:
            shutil.copy2(p,target/p.name)
            artifacts[str((target/p.name).relative_to(q))]=sha(target/p.name)
    if label=='R65-B1':shutil.copy2(origin/'replay',target/'replay')
    else:
        run('helper',['g++','-std=c++17','-O3','-DNDEBUG','-pthread','-I'+str(src/'common'),'-I'+str(src/'include'),'-I'+str(src/'ggml/include'),'-I'+str(src/'vendor'),str(q/'r65-source/moe-replay.cpp'),'-o',str(target/'replay'),'-L'+str(target),'-Wl,-rpath,$ORIGIN','-lllama-common','-lllama','-lggml','-lggml-base'])
    artifacts[str((target/'replay').relative_to(q))]=sha(target/'replay')
freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':parent['model'],'candidate_commit':'2e08b57e9997358109a993c845ce6b39eeb266fa','helper_source_sha256':sha(q/'r65-source/moe-replay.cpp'),'variants':{'R65-B1':'7fe450e19305b828c199d602c23a8337aaa1f03b unmodified B1 engine,existing frozen helper','R65-B4':head+' full unmodified upstream0.6.0-dev;canonical helper recompiled against candidate ABI'},'artifacts':artifacts,'inputs':protocol['inputs'],'protocol_sha256':sha(q/'r65-protocol.json'),'source_freeze_sha256':sha(q/'r65-source-freeze.json'),'sources':registered['sources'],'build_cache_sha256':sha(build/'CMakeCache.txt'),'compile_commands_sha256':sha(build/'compile_commands.json'),'compiler':subprocess.check_output(['g++','--version'],text=True),'relevant_build_environment':{k:v for k,v in os.environ.items()if k in ['CC','CXX','CFLAGS','CXXFLAGS','VULKAN_SDK']or k.startswith(('GGML_','VK_','AMD_','RADV_','MESA_','OMP_','LD_'))},'upstream_source_clean':True,'build_commands':'r65-build-v2-commands.json'}
(q/'r65-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
shutil.copy2(build/'CMakeCache.txt',q/'r65-CMakeCache.txt')
print('Current upstream source clean, B1 preserved, coherent libraries/helper frozen',flush=True)
