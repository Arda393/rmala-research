"""Direct mechanism tests separate from language-model training."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from .bank import ExactBank


def stream(seed,tau=.3,capacity=128,payload="residual",decay=.98):
    g=torch.Generator().manual_seed(seed)
    n,d=128,16
    keys=torch.rand(n,d,generator=g)+.05
    values=torch.randn(n,d,generator=g)
    # Deliberate overload: unique values exceed state key dimension eightfold.
    s=torch.zeros(d,d);z=torch.zeros(d)
    errors=[]; residuals=[]; written=[]
    bank=ExactBank(capacity)
    for k,v in zip(keys,values):
        s=decay*s+torch.outer(k,v);z=decay*z+k
        estimate=(k@s)/(k@z).clamp_min(1e-6)
        e=v-estimate
        error=e.norm()/v.norm().clamp_min(1e-6)
        errors.append(error);residuals.append(e)
        written.append(bool(error>tau))
        if error>tau:
            bank.write(k,e if payload=="residual" else v)
    base=(keys@s)/(keys@z).unsqueeze(-1).clamp_min(1e-6)
    corrected=[]
    for k,b in zip(keys,base):
        vals,scores=bank.query(k,8)
        r=torch.zeros_like(b) if vals is None else (scores.div(.1).softmax(0)[:,None]*vals).sum(0)
        corrected.append(b+r if payload=="residual" else r)
    corrected=torch.stack(corrected)
    final_error=(values-base).norm(dim=-1)/values.norm(dim=-1).clamp_min(1e-6)
    err=torch.stack(errors)
    bad=final_error>tau
    detected=torch.tensor(written)
    stats=dict(seed=seed,tau=tau,capacity=capacity,payload=payload,
               write_rate=sum(written)/n,baseline_mse=float((values-base).square().mean()),
               corrected_mse=float((values-corrected).square().mean()),
               detection_recall=float(detected[bad].float().mean()) if bad.any() else None,
               early_detection_recall=float(detected[:16][bad[:16]].float().mean()) if bad[:16].any() else None,
               bank_tensor_bytes=bank.tensor_bytes,bank_entries=bank.size,
               denominator_error_correlation=float(torch.corrcoef(torch.stack([keys@z,final_error]))[0,1]),
               write_final_error_correlation=float(torch.corrcoef(torch.stack([err,final_error]))[0,1]))
    return stats,torch.cat([keys,(keys@z).log()[:,None]],-1),base,corrected-base,values


def run(out,quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(17);torch.set_num_threads(2)
    started=time.perf_counter()
    records=[]
    seeds=range(3) if quick else range(5)
    for seed in seeds:
        for tau in ((.1,.3,.5) if quick else (.1,.2,.3,.5)):
            for cap in ((16,128) if quick else (16,64,128)):
                for payload in ("residual","raw"):
                    records.append(stream(seed,tau,cap,payload)[0])
    training=[stream(seed) for seed in range(10,14)]
    x=torch.cat([r[1] for r in training]);base=torch.cat([r[2] for r in training])
    correction=torch.cat([r[3] for r in training]);target=torch.cat([r[4] for r in training])
    gate=nn.Linear(17,1);opt=torch.optim.Adam(gate.parameters(),lr=.03)
    for _ in range(250):
        opt.zero_grad()
        a=gate(x).sigmoid()
        loss=(base+a*correction-target).square().mean()
        loss.backward();opt.step()
    testing=[stream(seed) for seed in range(100,105)]
    gate_results=[]
    with torch.no_grad():
        for r,x,b,c,y in testing:
            a=gate(x).sigmoid()
            for threshold in (.25,.5,.75,.9):
                hard=(a>=threshold).float()
                gate_results.append(dict(seed=r["seed"],threshold=threshold,
                                         soft_mse=float((b+a*c-y).square().mean()),
                                         hard_mse=float((b+hard*c-y).square().mean()),
                                         retrieval_rate=float(hard.mean())))
    # Constructive counterexample: exact first write cannot flag later forgetting.
    key=torch.tensor([1.,0.]);v=torch.tensor([1.,0.])
    s=torch.outer(key,v);z=key.clone()
    initial_error=float((v-key@s/(key@z)).norm())
    for _ in range(100):
        s=.98*s+torch.outer(key,torch.tensor([0.,1.]));z=.98*z+key
    later_error=float((v-key@s/(key@z)).norm())
    example=dict(initial_error=initial_error,later_error=later_error,
                 original_token_was_written=False,explanation="Perfect first reconstruction; later interfering writes cannot retroactively trigger its write")
    # A saved e=2-1.5=.5 becomes wrong when current base drifts to 1.
    stale=dict(value=2.,base_at_write=1.5,residual=.5,current_base=1.,corrected=1.5,error=.5)
    selected=[r for r in records if r["tau"]==.3 and r["capacity"]==128 and r["payload"]=="residual"]
    gate05=[r for r in gate_results if r["threshold"]==.5]
    mean=lambda rs,k:sum(r[k] for r in rs)/len(rs)
    failures=[]
    if mean(selected,"write_rate")>.05:
        failures.append("Default write rate exceeds proposed 2-5% budget on overload task")
    if later_error>.3 and initial_error<1e-6:
        failures.append("Write-time reconstruction does not detect later forgetting of originally well-represented tokens")
    failures.append("Dense residual and raw payloads require identical tensor bytes at equal entry count/dtype; compression is not implemented")
    if mean(gate05,"hard_mse")>1.01*mean(gate05,"soft_mse"):
        failures.append("Hard gate causes more than 1% mean MSE regression at threshold 0.5")
    failures.append("Stored residuals can become stale under state drift; no refresh/anchor mechanism yet")
    result=dict(production_ready=False,status="pilot_complete_research_gates_failed",seeds=list(seeds),quick=quick,
                default_mean_write_rate=mean(selected,"write_rate"),
                default_baseline_mse=mean(selected,"baseline_mse"),
                default_residual_mse=mean(selected,"corrected_mse"),
                gate_soft_mse=mean(gate05,"soft_mse"),gate_hard_mse=mean(gate05,"hard_mse"),
                gate_retrieval_rate=mean(gate05,"retrieval_rate"),
                counterexample=example,stale_residual_example=stale,failures=failures,
                elapsed_seconds=time.perf_counter()-started,ablation_runs=len(records))
    (out/"pilot.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"ablations.json").write_text(json.dumps(records,indent=2)+"\n")
    (out/"gate_sweep.json").write_text(json.dumps(gate_results,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="runs/diagnostics")
    p.add_argument("--quick",action="store_true")
    a=p.parse_args();run(a.out,a.quick)
