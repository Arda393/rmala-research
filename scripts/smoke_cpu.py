"""Very small end-to-end checks; no model-quality conclusion is drawn."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rmala.train import run

torch.set_num_threads(2)
summaries=[]
for variant in ("full","gla","rmala","surprise"):
    cfg=dict(model=dict(vocab_size=64,dim=16,heads=1,layers=1,variant=variant,
                        dropout=0.,capacity=8,topk=2,tau=.3),
             task="mqar",seed=29,steps=2,block=16,batch_size=1,lr=.001,warmup_steps=1)
    summaries.append(run(cfg,f"runs/cpu_smoke_{variant}",device="cpu"))
Path("runs/cpu_smokes.json").write_text(json.dumps(summaries,indent=2)+"\n")
