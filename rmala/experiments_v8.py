"""Shared cross-workload utility gates and offline routing attribution."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from torch import nn
from .budget_memory import PrefixBudget
from .experiments_v4 import mixed,test_context
from .experiments_v5 import projected,choose
from .experiments_v6 import INITIALIZATIONS,operation_counts
from .experiments_v7 import cache,train_gate,select_gate,evaluate_gate,probabilities


WORKLOADS=['dense_random','rare_details','correlated_dense','repeated_onehot']


def attribution(c,prob,threshold,hamming):
    """Evaluation-only labels; actual guard order, with mutually exclusive stages."""
    budget=PrefixBudget(.05)
    counts={n:0 for n in ['empty_bank','budget','precheck','post_reject','applied']}
    useful={n:0 for n in counts};gain={n:0. for n in counts};applied=[]
    for i,(p,d) in enumerate(zip(prob,c['distance'])):
        budget.advance()
        if not c['bank'].entries:stage='empty_bank'
        elif not budget.available:stage='budget'
        elif d>hamming:stage='precheck'
        else:
            assert budget.consume()
            stage='applied' if p>=threshold else 'post_reject'
        counts[stage]+=1;applied.append(stage=='applied')
        delta=float(c['base_error'][i]-c['out_error'][i])
        if bool(c['stored'][i]) and delta>1e-9:
            useful[stage]+=1;gain[stage]+=delta
    return dict(stage_counts=counts,useful_stored_by_stage=useful,
        useful_stored_mse_gain_sum_by_stage=gain,
        useful_stored_total=sum(useful.values()),attributed_attempts=budget.used),torch.tensor(applied)


def run(out,checkpoint):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    states=torch.load(checkpoint,map_location='cpu');models={}
    for init in INITIALIZATIONS:
        for method in ['fixed_context','positive_only']:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str((method,init))]);model.requires_grad_(False)
            models[method,init]=model
    count=0;choices=[];training=[];weights={}
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(name,s):mixed(name,s,mode) for name in WORKLOADS for s in list(range(1210,1218))+list(range(1220,1230))}
            selected=[]
            for (projection,init),model in models.items():
                cs={}
                for (name,s),b in bases.items():cs[name,s]=cache(projected(copy.deepcopy(b),name,s,'learned_context',model))
                pooled_cal=[cs[name,s] for name in WORKLOADS for s in range(1210,1218)]
                pooled_val=[cs[name,s] for name in WORKLOADS for s in range(1220,1230)]
                shared,log=train_gate(pooled_cal,init);shared_choice,grid=select_gate(pooled_cal,pooled_val,shared)
                similarity=choose(pooled_cal,pooled_val,True)
                info=dict(mode=mode,projection=projection,initialization=init)
                choices.append(dict(**info,scope='shared',utility=shared_choice,similarity=similarity,grid=grid))
                training.append(dict(**info,scope='shared',**log));weights[str((mode,projection,init,'shared'))]=shared.state_dict()
                local={}
                for name in WORKLOADS:
                    cal=[cs[name,s] for s in range(1210,1218)];val=[cs[name,s] for s in range(1220,1230)]
                    gate,log=train_gate(cal,init);choice,grid=select_gate(cal,val,gate);local[name]=(gate,choice)
                    choices.append(dict(**info,scope=name,utility=choice,grid=grid));training.append(dict(**info,scope=name,**log))
                    weights[str((mode,projection,init,name))]=gate.state_dict()
                selected.append((projection,init,model,shared,shared_choice,similarity,local))
                print(json.dumps(dict(selected=projection,init=init,mode=mode)),flush=True)
            # Tests created after every policy for this storage type is fixed.
            tests={(name,s):mixed(name,s,mode) for name in WORKLOADS for s in range(1300,1320)}
            for projection,init,model,shared,shared_choice,similarity,local in selected:
                for (name,seed),b in tests.items():
                    c=cache(projected(copy.deepcopy(b),name,seed,'learned_context',model))
                    for arm in ['shared_similarity','shared_utility','task_utility']:
                        if arm=='shared_similarity':
                            row=test_context(c,similarity,'constrained')
                            p=c['score'];t=similarity['similarity'];h=similarity['hamming']
                            row.update(gate_evaluations=0,gate_parameter_and_buffer_bytes=0,gate_linear_arithmetic_ops=0)
                        else:
                            gate,choice=(shared,shared_choice) if arm=='shared_utility' else local[name]
                            row=evaluate_gate(c,gate,choice);p=probabilities(gate,c);t=choice['threshold'];h=choice['hamming']
                        detail,mask=attribution(c,p,t,h)
                        assert detail['attributed_attempts']==row['attempted_reads']
                        assert int(mask.sum())==row['applied_corrections']
                        err=torch.where(mask,c['out_error'],c['base_error'])
                        assert abs(float(err.mean())-row['mse'])<1e-5
                        row.update(detail);row.update(operation_counts(row,True))
                        row.update(workload=name,mode=mode,projection=projection,initialization=init,seed=seed,arm=arm)
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                print(json.dumps(dict(mode=mode,projection=projection,init=init,rows=count)),flush=True)
    torch.save(weights,out/'utility_gates.pt');(out/'choices.json').write_text(json.dumps(choices,indent=2));(out/'training.json').write_text(json.dumps(training,indent=2))
    summary=dict(status='completed',rows=count,seconds=time.perf_counter()-start,production_ready=False,
        projection_checkpoint=str(checkpoint),initializations=INITIALIZATIONS,
        train_calibration_seeds=list(range(1210,1218)),validation_seeds=list(range(1220,1230)),test_seeds=list(range(1300,1320)),
        limitations=['Shared means one gate/threshold across workload contexts for each projection/storage; not one combined source bank.',
        'Task-specific gate is a diagnostic control requiring known workload; shared gates receive no workload label.',
        'Empty banks do not train a retrieval gate, because no retrieval decision exists there.',
        'Offline attribution uses target labels only after live routing and is not an oracle routing policy.',
        'Projection arithmetic estimates exclude gate/feature and full-model costs; no end-to-end FLOP claim.',
        'Synthetic context and frozen projections, no LM or 100M training.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--checkpoint',default='runs/v6_reference/projections.pt');a=p.parse_args();run(a.out,a.checkpoint)
