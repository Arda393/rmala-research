"""Frozen V14 with fixed alpha, matched and reselected thresholds."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import replay
from .utility_gate import UtilityGate
from .cheap_policy import CheapSketchRouter
from .nil_gate import match_targets
from .v21_gate import fine_cuts
from .v22_blend import blend

ALPHAS=[1.,.5,.25,.1]
NAMES=['dense_random','rare_details','correlated_dense','repeated_onehot']


def prepare(c,gate,h):
    p=probabilities(gate,c)
    eligible,reads,_=replay(c,p,0.,h)
    return dict(c=c,p=torch.tensor(p),eligible=eligible,reads=reads)


def select(cal,val,h):
    pool=[float(p) for z in cal for p,m in zip(z['p'],z['eligible']) if m]
    cuts=fine_cuts(pool);choices={};logs=[]
    for alpha in ALPHAS:
        errors=[(blend(z['c']['base'],z['c']['output'],alpha)-z['c']['y']).square().mean(-1) for z in val]
        grid=[]
        for t in cuts:
            safe=True;total=reads=accept=0
            for z,e in zip(val,errors):
                c=z['c'];mask=z['eligible']&(z['p']>=t)
                err=torch.where(mask,e,c['base_error'])
                safe=safe and safe_errors(c,err);total+=float(err.mean())
                reads+=z['reads'] if t<=1 else 0;accept+=int(mask.sum())
            grid.append(dict(threshold=t,hamming=h,safe=safe,mse=total/len(val),reads=reads,acceptances=accept))
        best=min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],-g['threshold']))
        choices[alpha]=best;logs.append(dict(alpha=alpha,selected=best,grid=grid))
    return choices,logs


def route(c,gate,t,h):
    router=CheapSketchRouter(c['bank'],gate,dict(threshold=t,hamming=h),h)
    ys=[];m=[]
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,_=router.query(q,b);ys.append(y);m.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    mask=torch.tensor(m);ys=torch.stack(ys)
    expected,reads,_=replay(c,probabilities(gate,c),t,h)
    assert torch.equal(mask,expected) and router.attempts==reads
    assert torch.allclose(ys,torch.where(mask[:,None],c['output'],c['base']),atol=1e-5,rtol=1e-5)
    assert router.bytes<=2048 and c['bank'].write_budget.used<=25
    return ys,mask,router


def metrics(c,ys,mask,router,alpha):
    actual=blend(c['base'],ys,alpha);err=(actual-c['y']).square().mean(-1)
    expected=torch.where(mask,(blend(c['base'],c['output'],alpha)-c['y']).square().mean(-1),c['base_error'])
    assert torch.allclose(err,expected,atol=1e-5,rtol=1e-5)
    delta=err-c['base_error'];harm=mask&(delta>1e-9);protected=~c['rare']|~c['stored'];correct=match_targets(c['ids'],c['selected_ids'])
    nb=float(c['base_error'][c['noisy']].mean());nm=float(err[c['noisy']].mean())
    return dict(mse=float(err.mean()),baseline_mse=float(c['base_error'].mean()),noisy_mse=nm,noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),
        subgroup_preservation_pass=safe_errors(c,err),attempted_reads=router.attempts if router else 0,
        applied_corrections=int(mask.sum()),useful_applications=int((mask&(delta < -1e-9)).sum()),harmful_acceptances=int(harm.sum()),
        harm_sum=float(delta[harm].double().sum()),benefit_sum=float((-delta[mask&(delta<0)]).double().sum()),
        worst_query_harm=max(0.,float(delta.max())),protected_harmful_acceptances=int((harm&protected).sum()),
        protected_harm_sum=float(delta[harm&protected].double().sum()),correct_identity_acceptances=int((mask&correct).sum()),
        wrong_identity_acceptances=int((mask&~correct).sum()),missing_acceptances=int((mask&~c['stored']).sum()),
        total_tensor_bytes=router.bytes if router else 0,writes=c['bank'].write_budget.used if router else 0,
        full_key_comparisons=router.comparisons if router else 0,post_linear_ops=router.gate_evaluations*288 if router else 0,
        post_static_bytes=708 if router else 0,blend_scalar_ops=int(mask.sum())*c['base'].shape[-1]*3 if 0<alpha<1 else 0)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    projections=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu');old=json.loads(ch.read_text());selected=[];logs=[]
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(projections[str(('positive_only',init))]);model.requires_grad_(False)
            gate=UtilityGate();gate.load_state_dict(weights[str(('raw_int8',init,1000))]);gate.requires_grad_(False)
            fixed=next(x['selected'] for x in old if x['mode']=='raw_int8' and x['initialization']==init and x['step']==1000)
            def make(n,s):return prepare(cache(projected(mixed(n,s,'raw_int8'),n,s,'learned_context',model)),gate,fixed['hamming'])
            cal=[make(n,s) for n in NAMES for s in range(5140,5156)]
            val=[make(n,s) for n in NAMES for s in range(5200,5300)]
            choices,grid=select(cal,val,fixed['hamming']);selected.append((init,model,gate,fixed,choices))
            logs.append(dict(initialization=init,fixed=fixed,alpha_choices=grid))
            print(json.dumps(dict(initialization=init,selected=choices)),flush=True);del cal,val
        (out/'choices.json').write_text(json.dumps(logs,indent=2));count=0
        with (out/'comparisons.jsonl').open('w') as f:
            for init,model,gate,fixed,choices in selected:
                for n in NAMES:
                    for seed in range(5400,5500):
                        c=cache(projected(mixed(n,seed,'raw_int8'),n,seed,'learned_context',model))
                        base=metrics(c,c['base'],torch.zeros(len(c['q']),dtype=torch.bool),None,0.)
                        base.update(initialization=init,workload=n,seed=seed,mode='raw_int8',policy='baseline',alpha=0.)
                        f.write(json.dumps(base)+'\n');count+=1;routes={}
                        for policy in ['fixed','selected']:
                            for alpha in ALPHAS:
                                t=fixed['threshold'] if policy=='fixed' else choices[alpha]['threshold']
                                if t not in routes:routes[t]=route(c,gate,t,fixed['hamming'])
                                row=metrics(c,*routes[t],alpha)
                                row.update(initialization=init,workload=n,seed=seed,mode='raw_int8',policy=policy,alpha=alpha,threshold=t)
                                f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==10800
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]},calibration=[5140,5155],validation=[5200,5299],test=[5400,5499],
        trained=False,production_ready=False,limitations=['Frozen V14 int8 gate/projection; alpha constant and not learned.',
        'Fixed threshold isolates alpha effect. Selected threshold separately tuned for each alpha on identical calibration/validation.',
        'Original strict subgroup tolerance retained; all choices saved before test. No risk-budget relaxation.',
        'Small alpha need not eliminate harmful events. Report magnitudes, worst harm, useful applications and net MSE together.',
        'Post-read blend does not reduce retrieval FLOPs; blend operation count excludes framework overhead.',
        'Three model initializations reuse contexts. Synthetic frozen banks; no real-text, 100M or full-model speed result.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
