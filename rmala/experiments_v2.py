"""Audit saturation and compare memory strategies under equal resource ceilings.

All gate tuning uses disjoint calibration/validation seeds. Test targets never
enter online routing. Dense retrieval used for calibration is costed separately
from actual held-out hard-gated retrieval. Results are mechanism tests, not LM BPB.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .budget_memory import BudgetMemory,PrefixBudget


def workload(name,seed,n=128,d=16):
    rng=torch.Generator().manual_seed(seed)
    keys=torch.rand(n,d,generator=rng)+.05
    values=torch.randn(n,d,generator=rng)
    rare=torch.zeros(n,dtype=torch.bool)
    if name=='elu_random': keys=F.elu(torch.randn(n,d,generator=rng))+1
    elif name=='orthogonal':
        keys=torch.eye(n);values=torch.randn(n,n,generator=rng)
    elif name=='repeated_onehot':
        keys=F.one_hot(torch.arange(n)%d,d).float()
        values=torch.randn(d,d,generator=rng)[torch.arange(n)%d]
    elif name=='correlated_dense':
        values=torch.randn(d,generator=rng)[None]+.01*values
    elif name=='rare_details':
        rare=torch.rand(n,generator=rng)<.1
        values=torch.randn(d,generator=rng)[None]+values*(rare[:,None]*4+.01)
    elif name!='dense_random': raise ValueError(name)
    return keys,values,rare


def audit(out,seeds):
    rows=[]
    for name in ['dense_random','elu_random','orthogonal','repeated_onehot','correlated_dense','rare_details']:
        for seed in seeds:
            k,v,rare=workload(name,seed);n,d=k.shape
            s=torch.zeros(d,v.shape[-1]);z=torch.zeros(d)
            errors=[];shares=[];absolute=[];reference_diff=[]
            for t in range(n):
                s=.98*s+torch.outer(k[t],v[t]);z=.98*z+k[t]
                rec=k[t]@s/(k[t]@z).clamp_min(1e-6)
                weights=(k[:t+1]@k[t])*.98**torch.arange(t,-1,-1)
                independent=(weights[:,None]*v[:t+1]).sum(0)/weights.sum().clamp_min(1e-6)
                reference_diff.append(float((rec-independent).abs().max()))
                errors.append(float((v[t]-rec).norm()/v[t].norm().clamp_min(1e-6)))
                absolute.append(float((v[t]-rec).square().mean().sqrt()))
                shares.append(float((k[t]@k[t])/(k[t]@z).clamp_min(1e-6)))
            cos=F.normalize(k,dim=-1)@F.normalize(k,dim=-1).T
            eps=torch.tensor(errors)
            rows.append(dict(workload=name,seed=seed,n=n,d=d,
                write_rate=float((eps>.3).float().mean()),written_indices=(eps>.3).nonzero().flatten().tolist(),
                epsilon_median=float(eps.median()),epsilon_p90=float(eps.quantile(.9)),
                epsilon=errors,self_weight=shares,reconstruction_rmse=absolute,
                mean_offdiagonal_key_cosine=float((cos.sum()-cos.diag().sum())/(n*(n-1))),
                final_self_weight=shares[-1],max_explicit_sum_difference=max(reference_diff),
                threshold_rates={str(t):float((eps>t).float().mean()) for t in [.1,.3,.5,.8,1.,1.2]},
                rare_write_rate=float((eps[rare]>.3).float().mean()) if rare.any() else None,
                ordinary_write_rate=float((eps[~rare]>.3).float().mean())))
    dimensional=[]
    for d in [8,16,32,64,128]:
        for seed in seeds:
            k,v,_=workload('dense_random',seed,d=d)
            s=torch.zeros(d,d);z=torch.zeros(d);written=0
            for ki,vi in zip(k,v):
                s=.98*s+torch.outer(ki,vi);z=.98*z+ki
                e=(vi-ki@s/(ki@z)).norm()/vi.norm()
                written+=int(e>.3)
            dimensional.append(dict(d=d,seed=seed,write_rate=written/len(k)))
    result=dict(controls=rows,dimension_sweep=dimensional,
        explanation='Positive dense keys overlap strongly. Normalization averages independent random values; self weight shrinks. Dimension alone is not a guarantee of separation.',
        unit='one write decision per source token in one abstract head; not percent of document text stored',
        code_check='Independent explicit weighted sum compared against recurrence at every position')
    (out/'saturation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


MODES={
    'raw_fp32':dict(payload='raw'),
    'residual_fp32':dict(payload='residual'),
    'residual_age':dict(payload='residual',half_life=32),
    'residual_anchor':dict(payload='anchored'),
    'raw_delayed':dict(payload='raw',candidates=4,delay=4),
    'residual_delayed':dict(payload='residual',candidates=4,delay=4),
    'raw_int8':dict(payload='raw',int8=True),
    'residual_int8':dict(payload='residual',int8=True),
}


def context(name,seed,mode,write_rate,byte_budget=2048,n=128):
    k,v,rare=workload(name,seed,n=n)
    d=k.shape[1];s=torch.zeros(d,d);z=torch.zeros(d)
    bank=BudgetMemory(d,d,byte_budget=byte_budget,write_rate=write_rate,read_rate=1,
                      **MODES[mode])
    began=time.perf_counter()
    for ki,vi in zip(k,v):
        s=.98*s+torch.outer(ki,vi);z=.98*z+ki
        read=lambda key:(key@s)/(key@z).clamp_min(1e-6)
        bank.observe(ki,vi,read(ki),read)
    build_seconds=time.perf_counter()-began
    order=torch.randperm(len(k),generator=torch.Generator().manual_seed(seed+70000))
    return dict(bank=bank,s=s,z=z,k=k[order],v=v[order],rare=rare[order],
                token_ids=order,build_seconds=build_seconds)


def calibration_cache(ctx):
    bank=ctx['bank'];features=[];bases=[];corrections=[]
    for key in ctx['k']:
        den=(key@ctx['z']).clamp_min(1e-6);base=key@ctx['s']/den
        features.append(bank.gate_features(key,den,base))
        bases.append(base);corrections.append(bank.retrieve(key,base,touch=False))
    return torch.stack(features),torch.stack(bases),torch.stack(corrections),ctx['v']


def train_gates(caches,penalties,steps=100):
    x,b,c,y=[torch.cat([a[i] for a in caches]) for i in range(4)]
    scale=(b-y).square().mean().clamp_min(1e-6)
    result=[]
    with torch.enable_grad():
        for penalty in penalties:
            torch.manual_seed(811)
            gate=nn.Linear(6,1)
            opt=torch.optim.Adam(gate.parameters(),lr=.04)
            for _ in range(steps):
                opt.zero_grad();alpha=gate(x).sigmoid()
                loss=(b+alpha*c-y).square().mean()/scale+penalty*alpha.mean()
                loss.backward();opt.step()
            result.append((penalty,gate))
    return result


def choose(gates,validation,read_rate):
    best=None
    for penalty,gate in gates:
        predictions=[gate(c[0]).sigmoid().flatten().detach() for c in validation]
        pooled=torch.cat(predictions)
        thresholds=sorted(set([.5]+[float(pooled.quantile(q)) for q in [.0,.5,.75,.9,.99]]))
        for threshold in thresholds:
            total_error=0;count=0;reads=0
            for pred,(_,base,corr,y) in zip(predictions,validation):
                budget=PrefixBudget(read_rate);mask=[]
                for alpha in pred:
                    budget.advance()
                    opened=bool(alpha>=threshold) and budget.consume()
                    mask.append(opened)
                active=torch.tensor(mask,dtype=torch.float32)
                # Empty bank corrections are zero; gate tuning cannot gain there.
                total_error+=float((base+active[:,None]*corr-y).square().sum())
                count+=y.numel();reads+=sum(mask)
            candidate=(total_error/count,reads,penalty,threshold,gate)
            if best is None or candidate[:4]<best[:4]: best=candidate
    return dict(validation_mse=best[0],penalty=best[2],threshold=best[3],gate=best[4])


def evaluate_context(ctx,choice,read_rate):
    bank=copy.deepcopy(ctx['bank']);bank.read_budget=PrefixBudget(read_rate)
    corrected=[];baseline=[];soft=[]
    start=time.perf_counter()
    for key in ctx['k']:
        den=(key@ctx['z']).clamp_min(1e-6);base=key@ctx['s']/den
        alpha=choice['gate'](bank.gate_features(key,den,base)).sigmoid().squeeze()
        corrected.append(bank.query(key,base,alpha,choice['threshold'],hard=True))
        baseline.append(base)
    query_seconds=time.perf_counter()-start
    baseline=torch.stack(baseline);corrected=torch.stack(corrected)
    errors=(corrected-ctx['v']).square().mean(-1)
    base_errors=(baseline-ctx['v']).square().mean(-1)
    # Dense soft path is a SEPARATE diagnostic; its searches are not hidden in
    # the timed hard path above. It has no read cap and is labelled accordingly.
    for key,base in zip(ctx['k'],baseline):
        den=(key@ctx['z']).clamp_min(1e-6)
        alpha=choice['gate'](bank.gate_features(key,den,base)).sigmoid().squeeze()
        soft.append(base+alpha*bank.retrieve(key,base,touch=False))
    rare=ctx['rare'];early=ctx['token_ids']<16
    result=dict(mse=float(errors.mean()),baseline_mse=float(base_errors.mean()),
        dense_soft_mse=float((torch.stack(soft)-ctx['v']).square().mean()),
        rare_mse=float(errors[rare].mean()) if rare.any() else None,
        rare_baseline_mse=float(base_errors[rare].mean()) if rare.any() else None,
        early_mse=float(errors[early].mean()),early_baseline_mse=float(base_errors[early].mean()),
        query_seconds=query_seconds,context_build_seconds=ctx['build_seconds'],
        persistent_state_bytes=4*(ctx['s'].numel()+ctx['z'].numel()),
        gate_penalty=choice['penalty'],gate_threshold=choice['threshold'],
        validation_mse=choice['validation_mse'],**bank.stats())
    if result['write_rate']>bank.write_budget.rate+1e-8 or result['retrieval_rate']>read_rate+1e-8:
        raise AssertionError('Prefix budgets violated')
    return result


def run(out,quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);started=time.perf_counter()
    calibration_seeds=[10,11] if quick else [10,11,12,13]
    validation_seeds=[20] if quick else [20,21]
    test_seeds=[100,101] if quick else [100,101,102,103,104]
    audit_result=audit(out,[0,1] if quick else [0,1,2,3,4])
    write_rates=[.01,.02,.05,.1];read_rates=[.01,.02,.05,.1]
    workloads=['dense_random','rare_details','correlated_dense']
    penalties=[0.,.2,1.] if quick else [0.,.05,.2,.5,1.]
    records=[];choices=[]
    with torch.no_grad():
        for name in workloads:
            for mode in MODES:
                for wr in write_rates:
                    cal=[calibration_cache(context(name,s,mode,wr)) for s in calibration_seeds]
                    val=[calibration_cache(context(name,s,mode,wr)) for s in validation_seeds]
                    gates=train_gates(cal,penalties,steps=50 if quick else 100)
                    contexts=[context(name,s,mode,wr) for s in test_seeds]
                    for rr in read_rates:
                        choice=choose(gates,val,rr)
                        choice_row=dict(workload=name,mode=mode,write_budget=wr,read_budget=rr,
                            penalty=choice['penalty'],threshold=choice['threshold'],validation_mse=choice['validation_mse'])
                        choices.append(choice_row)
                        for seed,ctx in zip(test_seeds,contexts):
                            row=dict(workload=name,mode=mode,seed=seed,write_budget=wr,read_budget=rr,
                                     **evaluate_context(ctx,choice,rr))
                            records.append(row)
                            with (out/'comparisons.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(progress=name+'/'+mode,rows=len(records))),flush=True)
    summary=dict(status='completed',production_ready=False,quick=quick,
        calibration_seeds=calibration_seeds,validation_seeds=validation_seeds,test_seeds=test_seeds,
        configurations=len(choices),test_runs=len(records),byte_budget_per_head=2048,
        write_rates=write_rates,read_rates=read_rates,elapsed_seconds=time.perf_counter()-started,
        caveats=['Synthetic fixed-key/value mechanism tasks, not learned LM BPB or canonical NIAH/MQAR.',
                 'Equal persistent tensor byte ceilings include candidate buffers, anchors and tensor metadata; Python/allocator overhead is excluded.',
                 'Strict event ceilings are equal, actual reads/writes/comparisons can differ; no equal-FLOP claim.',
                 'Dense calibration and dense soft diagnostics perform additional retrieval; only held-out hard path query_seconds measures actual skipped scans.',
                 'Anchored residual reconstructs the stored raw value and is not a free residual-only fix.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'gate_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    if (Path(a.out)/'comparisons.jsonl').exists(): raise FileExistsError('Choose a fresh output directory')
    run(a.out,a.quick)
