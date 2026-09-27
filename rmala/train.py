"""Bounded training/evaluation runner. Production runs require a passed pilot."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import LanguageModel
from .synthetic import batch
from .data import Corpus, metrics, digest


def evaluate(model,corpus,block,device,max_docs=None,hard=None):
    model.eval()
    nll=tokens=nbytes=0
    rows=corpus.docs[corpus.docs["eval"]==1]
    if max_docs is not None:
        rows=rows[:max_docs]
    with torch.no_grad():
        for row in rows:
            for x,y in corpus.chunks(row,block):
                logits=model(torch.as_tensor(x,device=device)[None],hard=hard)
                target=torch.as_tensor(y,device=device)
                nll+=float(F.cross_entropy(logits[0].float(),target,reduction="sum"))
                tokens+=len(y)
            nbytes+=int(row["bytes"])
    result=metrics(nll,tokens,nbytes)
    result.update(documents=len(rows),context_policy="disjoint chunks, reset per chunk; every token scored once; EOS start context",
                  subset_eval=bool(max_docs is not None and max_docs<len(corpus.docs[corpus.docs["eval"]==1])))
    return result


def run(config,out,steps=None,device=None,resume=None):
    cfg=json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else config
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg.get("seed",17)); np.random.seed(cfg.get("seed",17))
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    mc=cfg["model"]
    if mc.get("dim",768)>=512:
        gate=cfg.get("pilot_report")
        if not gate or not json.loads(Path(gate).read_text()).get("production_ready",False):
            raise RuntimeError("100M-scale launch blocked: pilot scientific/engineering gates have not passed")
    model=LanguageModel(**mc).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get("lr",6e-4),weight_decay=.1)
    corpus=Corpus(cfg["corpus"]) if cfg.get("corpus") else None
    start_step=0
    if resume:
        state=torch.load(resume,map_location=device)
        if state["config"]!=cfg:
            raise ValueError("Resume config mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step=state["step"]
    count=steps or cfg.get("steps",100)
    accum=cfg.get("gradient_accumulation",1)
    block=cfg.get("block",32)
    effective_batch=cfg.get("batch_size",2)
    if corpus and effective_batch!=1:
        raise ValueError("Document-isolated reference LM loader requires batch_size=1; use gradient accumulation")
    rows=corpus.docs[corpus.docs["eval"]==0] if corpus else None
    # Deterministic bounded sampler. Full corpus-epoch traversal must use a
    # production packed/document-aware loader; never imply complete 8GB coverage.
    started=time.perf_counter()
    elapsed_prior=state.get("elapsed_seconds",0.) if resume else 0.
    tokens_seen=state.get("tokens_seen",0) if resume else 0
    logpath=out/"metrics.jsonl"
    if not resume and logpath.exists():
        raise FileExistsError("Output already contains a run; select a new output or resume")
    for step in range(start_step,count):
        model.train(); optimizer.zero_grad(set_to_none=True)
        warmup=max(1,cfg.get("warmup_steps",10))
        ratio=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,count-warmup)))
        for group in optimizer.param_groups:
            group["lr"]=cfg.get("lr",6e-4)*ratio
        loss_sum=0
        for micro in range(accum):
            serial=step*accum+micro
            if corpus:
                rng=np.random.default_rng(cfg.get("seed",17)+serial)
                # Token-weighted document sampling avoids over-weighting short docs.
                weights=rows["count"].astype(float); weights/=weights.sum()
                row=rows[int(rng.choice(len(rows),p=weights))]
                tok=corpus.document(row)
                offset=int(rng.integers(max(1,len(tok)-block+1)))
                target=tok[offset:offset+block]
                first=2 if offset==0 else tok[offset-1]
                inputs=np.concatenate(([first],target[:-1]))
                x=torch.as_tensor(inputs)[None]; y=torch.as_tensor(target)[None]
            else:
                x,y=batch(cfg.get("task","mqar"),effective_batch,block,
                          cfg.get("seed",17)+serial,vocab_size=mc["vocab_size"])
            x,y=x.to(device),y.to(device)
            use_amp=device.startswith("cuda") and cfg.get("bf16",False)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=torch.bfloat16,enabled=use_amp):
                logits=model(x)
                loss=F.cross_entropy(logits.float().reshape(-1,mc["vocab_size"]),y.reshape(-1),ignore_index=-100)
                costs=[block.attn.gate_cost for block in model.blocks
                       if getattr(block.attn,'gate_cost',None) is not None]
                if costs:
                    loss=loss+cfg.get('gate_cost_weight',0.)*torch.stack(costs).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            (loss/accum).backward()
            loss_sum+=float(loss.detach())/accum
            tokens_seen+=int((y!=-100).sum())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed=elapsed_prior+time.perf_counter()-started
        record=dict(step=step+1,loss=loss_sum,elapsed_seconds=elapsed,
                    gpu_hours=elapsed/3600 if device.startswith("cuda") else None,
                    measured_flops=None,flops_note="not measured; no 6NT estimate for memory routing",
                    tokens_seen=tokens_seen,memory=model.memory_stats())
        with logpath.open("a") as f:
            f.write(json.dumps(record)+"\n")
    model.eval()
    evaluation={}
    if corpus:
        evaluation=evaluate(model,corpus,block,device,max_docs=cfg.get("eval_docs",8))
    else:
        for hard in (False,True):
            correct=total=0
            with torch.no_grad():
                for seed in range(10000,10008):
                    x,y=batch(cfg.get("task","mqar"),effective_batch,block,seed,mc["vocab_size"])
                    pred=model(x.to(device),hard=hard).argmax(-1).cpu()
                    valid=y!=-100
                    correct+=int((pred[valid]==y[valid]).sum()); total+=int(valid.sum())
            evaluation["hard" if hard else "soft"]=dict(accuracy=correct/total,queries=total)
    elapsed=elapsed_prior+time.perf_counter()-started
    ckpt=dict(config=cfg,step=count,model=model.state_dict(),optimizer=optimizer.state_dict(),
              rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
              tokens_seen=tokens_seen,elapsed_seconds=elapsed)
    torch.save(ckpt,out/"checkpoint.tmp")
    (out/"checkpoint.tmp").replace(out/"checkpoint.pt")
    summary=dict(config=cfg,parameters=sum(p.numel() for p in model.parameters()),device=device,
                 torch_version=torch.__version__,steps=count,elapsed_seconds=elapsed,evaluation=evaluation,
                 corpus_manifest_sha256=digest(Path(cfg["corpus"])/"manifest.json") if corpus else None,
                 status="smoke_complete",production_ready=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    p.add_argument("--steps",type=int);p.add_argument("--device");p.add_argument("--resume")
    a=p.parse_args();run(a.config,a.out,a.steps,a.device,a.resume)
