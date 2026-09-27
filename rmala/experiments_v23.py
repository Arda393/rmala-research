"""Learned per-query alpha against fixed and optimal-global controls."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache
from .experiments_v22 import prepare,route,NAMES
from .utility_gate import UtilityGate
from .nil_gate import match_targets
from .v21_gate import fine_cuts
from .v23_alpha import train,predict,tensor_blend

ARMS=['full','half','scalar','query']


def select(cal,val,head,scalar,h):
    cuts=fine_cuts([float(p) for z in cal for p,m in zip(z['p'],z['eligible']) if m]);choices={};logs=[]
    for arm in ARMS:
        errors=[]
        for z in val:
            c=z['c'];a=predict(c,z['eligible'],arm,head,scalar)
            errors.append((tensor_blend(c['base'],c['output'],a)-c['y']).square().mean(-1))
        grid=[]
        for t in cuts:
            safe=True;total=reads=accepted=0
            for z,e in zip(val,errors):
                c=z['c'];mask=z['eligible']&(z['p']>=t);err=torch.where(mask,e,c['base_error'])
                safe=safe and safe_errors(c,err);total+=float(err.mean());reads+=z['reads'] if t<=1 else 0;accepted+=int(mask.sum())
            grid.append(dict(threshold=t,hamming=h,safe=safe,mse=total/len(val),reads=reads,acceptances=accepted))
        best=min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],-g['threshold']))
        choices[arm]=best;logs.append(dict(arm=arm,selected=best,grid=grid))
    return choices,logs


def metrics(c,ys,mask,router,arm,head,scalar):
    if arm=='baseline':a=torch.zeros(len(mask))
    else:a=predict(c,mask,arm,head,scalar,live_values=ys)
    cached=torch.zeros_like(a) if arm=='baseline' else predict(c,mask,arm,head,scalar)
    assert torch.allclose(a,cached,atol=1e-5,rtol=1e-5)
    actual=tensor_blend(c['base'],ys,a)
    expected=tensor_blend(c['base'],c['output'],cached)
    assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-5)
    err=(actual-c['y']).square().mean(-1);delta=err-c['base_error'];harm=mask&(delta>1e-9);protected=~c['rare']|~c['stored']
    correct=match_targets(c['ids'],c['selected_ids']);nb=float(c['base_error'][c['noisy']].mean());nm=float(err[c['noisy']].mean())
    return dict(mse=float(err.mean()),baseline_mse=float(c['base_error'].mean()),noisy_mse=nm,noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),
        subgroup_preservation_pass=safe_errors(c,err),attempted_reads=router.attempts if router else 0,applied_corrections=int(mask.sum()),
        effective_corrections=int((mask&(a>0)).sum()),useful_applications=int((mask&(delta < -1e-9)).sum()),harmful_acceptances=int(harm.sum()),
        harm_sum=float(delta[harm].double().sum()),benefit_sum=float((-delta[mask&(delta<0)]).double().sum()),worst_query_harm=max(0.,float(delta.max())),
        protected_harmful_acceptances=int((harm&protected).sum()),protected_harm_sum=float(delta[harm&protected].double().sum()),
        correct_identity_acceptances=int((mask&correct).sum()),wrong_identity_acceptances=int((mask&~correct).sum()),
        alpha_sum=float(a.double().sum()),correct_alpha_sum=float(a[mask&correct].double().sum()),wrong_alpha_sum=float(a[mask&~correct].double().sum()),
        low_alpha_acceptances=int((mask&(a<.01)).sum()),total_tensor_bytes=router.bytes if router else 0,
        writes=c['bank'].write_budget.used if router else 0,full_key_comparisons=router.comparisons if router else 0,
        post_linear_ops=router.gate_evaluations*288 if router else 0,alpha_linear_ops=int(mask.sum())*288 if arm=='query' else 0,
        post_static_bytes=708 if router else 0,alpha_static_bytes=708 if arm=='query' else 4 if arm=='scalar' else 0,
        blend_scalar_ops=int(mask.sum())*c['base'].shape[-1]*3 if arm in ['half','scalar','query'] else 0)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    projections=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu');old=json.loads(ch.read_text());selected=[];logs=[];saved={}
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(projections[str(('positive_only',init))]);model.requires_grad_(False)
            gate=UtilityGate();gate.load_state_dict(weights[str(('raw_int8',init,1000))]);gate.requires_grad_(False)
            fixed=next(x['selected'] for x in old if x['mode']=='raw_int8' and x['initialization']==init and x['step']==1000)
            def make(n,s):return prepare(cache(projected(mixed(n,s,'raw_int8'),n,s,'learned_context',model)),gate,fixed['hamming'])
            training=[make(n,s) for n in NAMES[:2] for s in range(5600,5728)]
            head,scalar,info=train(training,init);saved[str(init)]=head.state_dict();del training
            cal=[make(n,s) for n in NAMES for s in range(5740,5756)]
            val=[make(n,s) for n in NAMES for s in range(5800,5900)]
            choices,grid=select(cal,val,head,scalar,fixed['hamming']);selected.append((init,model,gate,head,scalar,fixed,choices))
            logs.append(dict(initialization=init,fixed=fixed,training=info,choices=grid))
            print(json.dumps(dict(initialization=init,scalar=scalar,selected=choices)),flush=True);del cal,val
        torch.save(saved,out/'alpha_heads.pt');(out/'choices.json').write_text(json.dumps(logs,indent=2));count=0
        with (out/'comparisons.jsonl').open('w') as f:
            for init,model,gate,head,scalar,fixed,choices in selected:
                for n in NAMES:
                    for seed in range(6000,6100):
                        c=cache(projected(mixed(n,seed,'raw_int8'),n,seed,'learned_context',model));routes={}
                        base=metrics(c,c['base'],torch.zeros(len(c['q']),dtype=torch.bool),None,'baseline',head,scalar)
                        base.update(initialization=init,workload=n,seed=seed,mode='raw_int8',policy='baseline',arm='baseline')
                        f.write(json.dumps(base)+'\n');count+=1
                        for policy in ['fixed','selected']:
                            for arm in ARMS:
                                t=fixed['threshold'] if policy=='fixed' else choices[arm]['threshold']
                                if t not in routes:routes[t]=route(c,gate,t,fixed['hamming'])
                                row=metrics(c,*routes[t],arm,head,scalar)
                                row.update(initialization=init,workload=n,seed=seed,mode='raw_int8',policy=policy,arm=arm,threshold=t)
                                f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==10800
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]},training=[5600,5727],calibration=[5740,5755],validation=[5800,5899],test=[6000,6099],
        production_ready=False,limitations=['V14 retrieval/gate frozen; only alpha head trained. Same 8 observable inputs, no inference labels.',
        'Alpha head 8-16-1 sigmoid starts at 0.5; 1000 Adam steps on direct blended-output MSE over training read-eligible candidates.',
        'Scalar control is the analytic training-MSE optimum over the same candidates, not query-dependent.',
        'Fixed policy isolates contribution changes; selected policy tunes threshold for each arm with unchanged subgroup constraints.',
        'Extra head 708 static bytes and 288 linear operations per accepted candidate; post-read blending saves no search FLOPs.',
        'Alpha rounded to 5 decimals for query inference; near-zero use is reported and not counted as success.',
        'Synthetic frozen banks, shared test contexts across initializations; no real-text or 100M claim.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
