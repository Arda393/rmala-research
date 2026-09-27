"""Small execution checks, not a trained-model benchmark or a superiority claim."""
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rmala.train import run

torch.set_num_threads(2)
subprocess.run([sys.executable,'scripts/source_manifest.py'],check=True)
summaries=[]
for variant in ("full","gla","rmala","surprise"):
    cfg=dict(model=dict(vocab_size=256,dim=64,heads=2,layers=1,variant=variant,
                        dropout=0.,capacity=32,topk=8,tau=.3),
             task="mqar",seed=17,steps=30,block=32,batch_size=2,lr=.001,warmup_steps=5)
    summaries.append(run(cfg,f"runs/pilot_{variant}"))
Path("runs/model_pilots.json").write_text(json.dumps(summaries,indent=2)+"\n")

# Independent engineering check using an existing, pinned read-only dependency
# checkout. No modification/upgrade of shared environments is performed.
deps=Path('/arf/scratch/YOURUSER/mercanset_arch_bench_0p8b/third_party')
fla=deps/'flash-linear-attention'
expected='34682796dcd6ac462952b0485338e6e564eacd02'
try:
    revision=subprocess.check_output(['git','-C',str(fla),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(fla),'status','--porcelain'],text=True).strip()
    if revision!=expected or dirty:
        raise RuntimeError('FLA checkout changed from the recorded clean revision')
    env=dict(os.environ)
    env['PYTHONPATH']=os.pathsep.join([str(Path.cwd()),str(deps/'pydeps_min'),str(fla)])
    env['APPTAINERENV_PYTHONPATH']=env['PYTHONPATH']
    env['TRITON_CACHE_DIR']=str(Path.cwd()/'runs'/'triton_cache')
    result=subprocess.run(['apptainer','exec','--nv','--bind',
        '/arf/scratch/YOURUSER:/arf/scratch/YOURUSER',
        '/arf/scratch/YOURUSER/NedoFormer_V2/containers/train.sif',
        'python','-u','scripts/fla_probe.py'],env=env,timeout=600)
    if result.returncode:
        raise RuntimeError(f'FLA probe exit code {result.returncode}')
except Exception as exc:
    Path('runs/fla_parity.json').write_text(json.dumps(dict(status='FAIL',error=str(exc)),indent=2)+'\n')
    print('FLA engineering gate failed:',exc,flush=True)
