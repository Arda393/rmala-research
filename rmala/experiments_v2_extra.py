"""Active-read and capacity-binding controls for v2; no gate training/tuning.

The main learned gate can rationally choose zero reads. These controls isolate
retrieval, aging and quantization rather than calling zero-read equality a win.
Stored-key membership is used ONLY to stratify final metrics, never for routing.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from .budget_memory import PrefixBudget
from .experiments_v2 import context,MODES


def evaluate(ctx,read_rate,topk):
    bank=copy.deepcopy(ctx['bank']);bank.read_budget=PrefixBudget(read_rate)
    bank.topk=topk
    stored={int(e['meta'][0])-1 for e in bank.entries}
    out=[];base_values=[];opened=[]
    start=time.perf_counter()
    for key in ctx['k']:
        base=(key@ctx['s'])/(key@ctx['z']).clamp_min(1e-6)
        before=bank.queries
        out.append(bank.query(key,base,alpha=1.,hard=True))
        base_values.append(base);opened.append(bank.queries>before)
    seconds=time.perf_counter()-start
    prediction=torch.stack(out);base=torch.stack(base_values);target=ctx['v']
    error=(prediction-target).square().mean(-1)
    base_error=(base-target).square().mean(-1)
    member=torch.tensor([int(i) in stored for i in ctx['token_ids']])
    active=torch.tensor(opened,dtype=torch.bool)
    read_members=member&active
    rare=ctx['rare']
    def mean_selected(value,mask): return float(value[mask].mean()) if mask.any() else None
    return dict(mse=float(error.mean()),baseline_mse=float(base_error.mean()),
        stored_queries=int(member.sum()),stored_and_read=int(read_members.sum()),
        stored_read_mse=mean_selected(error,read_members),
        stored_read_baseline_mse=mean_selected(base_error,read_members),
        unstored_read_mse=mean_selected(error,(~member)&active),
        unstored_read_baseline_mse=mean_selected(base_error,(~member)&active),
        rare_mse=mean_selected(error,rare),rare_baseline_mse=mean_selected(base_error,rare),
        query_seconds=seconds,context_build_seconds=ctx['build_seconds'],
        current_entries=len(bank.entries),**bank.stats())


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'comparisons.jsonl').exists(): raise FileExistsError('Choose a fresh output')
    torch.set_num_threads(2);start=time.perf_counter();count=0
    with torch.no_grad():
        for name in ['dense_random','rare_details','correlated_dense']:
            for mode in MODES:
                for byte_budget in [1024,2048]:
                    for seed in [100,101,102,103,104]:
                        ctx=context(name,seed,mode,.05,byte_budget,n=512)
                        for topk in [1,8]:
                            for read_rate in [.05,1.]:
                                result=dict(workload=name,mode=mode,seed=seed,n=512,write_budget=.05,
                                    read_budget=read_rate,topk=topk,policy='always-open subject to prefix budget',
                                    **evaluate(ctx,read_rate,topk))
                                with (out/'comparisons.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
                                count+=1
                print(json.dumps(dict(progress=name+'/'+mode,rows=count)),flush=True)
    result=dict(status='completed',test_runs=count,seeds=[100,101,102,103,104],n=512,
        byte_budgets=[1024,2048],write_budget=.05,read_budgets=[.05,1.],topk=[1,8],
        elapsed_seconds=time.perf_counter()-start,
        caveats=['Active-read diagnostic, not the validation-selected production policy.',
                 'The 100% read setting intentionally spends more compute to isolate retrieval error.',
                 'Stored membership is used for reporting only; no oracle information enters routing.',
                 '512 tokens makes the same byte ceiling bind for fp32 and int8 at 5% writes.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    run(p.parse_args().out)
