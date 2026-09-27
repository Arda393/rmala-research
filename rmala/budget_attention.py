import torch
from .budget_memory import BudgetMemory


def forward(module,x,q,k,v,g,hard):
    b,t,h,d=q.shape
    s=q.new_zeros(b,h,d,d);z=q.new_zeros(b,h,d)
    banks=[[BudgetMemory(d,d,**module.budget_options) for _ in range(h)] for _ in range(b)]
    outputs=[];alphas=[]
    for i in range(t):
        decay=g[:,i].exp()
        s=s*decay[...,None]+k[:,i,:, :,None]*v[:,i].float()[:,:,None,:]
        z=z*decay+k[:,i]
        rows=[]
        for bi in range(b):
            heads=[]
            for hi in range(h):
                bank=banks[bi][hi]
                def state_read(key):
                    return (key@s[bi,hi])/(key@z[bi,hi]).clamp_min(1e-6)
                base=state_read(q[bi,i,hi]);recon=state_read(k[bi,i,hi])
                bank.observe(k[bi,i,hi],v[bi,i,hi],recon,state_read)
                den=q[bi,i,hi]@z[bi,hi]
                features=bank.gate_features(q[bi,i,hi],den,base)
                alpha=module.budget_gate(features.to(module.budget_gate.weight.dtype)).sigmoid().squeeze()
                alphas.append(alpha)
                heads.append(bank.query(q[bi,i,hi],base,alpha,module.hard_threshold,hard))
            rows.append(torch.stack(heads))
        outputs.append(torch.stack(rows))
    module.gate_cost=torch.stack(alphas).mean()
    all_stats=[bank.stats() for row in banks for bank in row]
    module.stats={key:sum(item[key] for item in all_stats)/(b*h)
                  for key in ['write_rate','retrieval_rate','writes','queries','comparisons','candidate_checks','delayed_writes']}
    module.stats.update(peak_tensor_bytes=sum(item['peak_tensor_bytes'] for item in all_stats),
                        byte_budget=sum(item['byte_budget'] for item in all_stats),
                        alpha_mean=float(module.gate_cost.detach()),budgeted=True)
    return module.proj(torch.stack(outputs,1).reshape(b,t,-1).to(x.dtype))
