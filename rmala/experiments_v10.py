"""Fresh-data pre-retrieval selection, retaining the frozen V8 post gate."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .pre_gate import pre_features,PreRouter
from .utility_gate import UtilityGate
from .rejection import signature
from .budget_memory import PrefixBudget
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities,evaluate_gate


def prepare(c,post):
    cache(c);c['post_prob']=probabilities(post,c)
    codes=[signature(e['key']) for e in c['bank'].entries]
    c['pre_features']=torch.stack([pre_features(q,b,codes,len(codes)/max(1,c['bank'].capacity)) for q,b in zip(c['q'],c['base'])])
    return c


def train(cs,choice,init):
    torch.manual_seed(init+10000);gate=UtilityGate();active=[c for c in cs if c['bank'].entries]
    x=torch.cat([c['pre_features'] for c in active]);labels=torch.cat([
        ((torch.tensor(c['post_prob'])>=choice['threshold'])&(c['out_error']<c['base_error']-1e-9)&
         (torch.tensor(c['distance'])<=choice['hamming'])).float() for c in active])
    gate.mean.copy_(x.mean(0));gate.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(gate.parameters(),lr=.01);rng=torch.Generator().manual_seed(init+11000)
    # Unweighted BCE estimates the probability of a useful accepted read.
    # Target values only construct training labels, never inference features.
    with torch.enable_grad():
        for _ in range(300):
            ids=torch.randperm(len(x),generator=rng)[:512];loss=F.binary_cross_entropy_with_logits(gate(x[ids]),labels[ids])
            opt.zero_grad();loss.backward();opt.step()
    return gate.requires_grad_(False),dict(steps=300,positive_fraction=float(labels.mean()),loss=float(loss.detach()))


def probs(g,c):return [round(float(g(x).sigmoid()),5) for x in c['pre_features']]


def replay(c,p,t,choice):
    budget=PrefixBudget(.05);mask=[]
    for pre,post,d in zip(p,c['post_prob'],c['distance']):
        budget.advance()
        read=bool(c['bank'].entries) and pre>=t and d<=choice['hamming'] and budget.consume()
        mask.append(read and post>=choice['threshold'])
    mask=torch.tensor(mask);return mask,budget.used,torch.where(mask,c['out_error'],c['base_error'])


def select(cal,val,g,choice):
    pool=torch.tensor([p for c in cal for p in probs(g,c)])
    thresholds=sorted(set([0.,.25,.5,.75,.9,1.00001]+[float(pool.quantile(q)) for q in [.5,.75,.9,.95,.99]]))
    vp=[probs(g,c) for c in val];grid=[]
    for t in thresholds:
        rs=[replay(c,p,t,choice) for c,p in zip(val,vp)]
        grid.append(dict(threshold=t,safe=all(safe_errors(c,r[2]) for c,r in zip(val,rs)),
                         mse=sum(float(r[2].mean()) for r in rs)/len(rs),reads=sum(r[1] for r in rs)))
    best=min([r for r in grid if r['safe']],key=lambda r:(r['mse'],r['reads'],-r['threshold']))
    return best,grid


def evaluate(c,post,choice,pre,t):
    router=PreRouter(c['bank'],post,choice,pre,t);ys=[];applied=[]
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,_=router.query(q,b);ys.append(y);applied.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    mask,reads,err=replay(c,probs(pre,c),t,choice)
    actual=(torch.stack(ys)-c['y']).square().mean(-1)
    assert torch.equal(mask,torch.tensor(applied)) and reads==router.attempts
    assert torch.allclose(err,actual,atol=1e-5,rtol=1e-5)
    return dict(mse=float(actual.mean()),baseline_mse=float(c['base_error'].mean()),
        noisy_mse=float(actual[c['noisy']].mean()),noisy_baseline_mse=float(c['base_error'][c['noisy']].mean()),
        subgroup_preservation_pass=safe_errors(c,actual),attempted_reads=reads,applied_corrections=int(mask.sum()),
        harmful_acceptances=int((mask&(actual>c['base_error']+1e-9)).sum()),
        pre_evaluations=router.pre_evaluations,extra_sketch_comparisons=router.extra_sketch_comparisons,
        pre_linear_arithmetic_ops=router.pre_evaluations*288,pre_static_bytes=pre.tensor_bytes,
        post_evaluations=router.gate_evaluations,total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v8_reference/utility_gates.pt');ch=Path('runs/v8_reference/choices.json')
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu');choices=json.loads(ch.read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];logs=[];saved={};count=0
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in names for s in list(range(1410,1418))+list(range(1420,1430))}
            selected=[]
            for init in [41001,41002,41003]:
                model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))])
                post=UtilityGate();post.load_state_dict(weights[str((mode,'positive_only',init,'shared'))])
                choice=next(x['utility'] for x in choices if x['mode']==mode and x['projection']=='positive_only' and x['initialization']==init and x['scope']=='shared')
                cs={(n,s):prepare(projected(copy.deepcopy(b),n,s,'learned_context',model),post) for (n,s),b in bases.items()}
                cal=[cs[n,s] for n in names for s in range(1410,1418)];val=[cs[n,s] for n in names for s in range(1420,1430)]
                pre,log=train(cal,choice,init);best,grid=select(cal,val,pre,choice)
                logs.append(dict(mode=mode,initialization=init,training=log,selected=best,grid=grid,post=choice))
                saved[str((mode,init))]=pre.state_dict();selected.append((init,model,post,choice,pre,best['threshold']))
                print(json.dumps(dict(selected=mode,init=init,threshold=best['threshold'])),flush=True)
            # Holdout contexts are created only after every policy in this mode is fixed.
            tests={(n,s):mixed(n,s,mode) for n in names for s in range(1500,1520)}
            for init,model,post,choice,pre,t in selected:
                for (n,s),b in tests.items():
                    c=prepare(projected(copy.deepcopy(b),n,s,'learned_context',model),post)
                    for arm in ['frozen_post','pre_and_post']:
                        row=evaluate_gate(c,post,choice) if arm=='frozen_post' else evaluate(c,post,choice,pre,t)
                        row.update(workload=n,mode=mode,initialization=init,seed=s,arm=arm)
                        assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                print(json.dumps(dict(mode=mode,init=init,rows=count)),flush=True)
    torch.save(saved,out/'pre_gates.pt');(out/'choices.json').write_text(json.dumps(logs,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]},production_ready=False,
        train_seeds=list(range(1410,1418)),validation_seeds=list(range(1420,1430)),test_seeds=list(range(1500,1520)),
        limitations=['Frozen V8 post gate, positive_only projections only; synthetic frozen banks, not LM training.',
        'Pre gate adds 708 static bytes and 288 linear arithmetic operations per populated query plus features and sketch scan.',
        'Parent repeats sketch precheck; extra work explicitly counted. No speed or full FLOP claim.',
        'No candidate, full retrieval scores, identity, membership or targets enter pre gate inference.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
