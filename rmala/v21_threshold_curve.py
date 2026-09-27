"""Exact score-breakpoint audit; no training or policy selection for deployment."""
import bisect


def sweep(events,group_sizes,baseline_sum,queries,attempts):
    events=sorted(events,key=lambda e:e['score'],reverse=True)
    sums={g:0. for g in group_sizes};bad=set();counts=dict(accepted=0,useful=0,harmful=0,correct=0,wrong=0,missing=0)
    delta=benefit=harm=worst=0.;rows=[]
    def row(t):
        return dict(threshold=t,**counts,benefit_sum=benefit,harm_sum=harm,net_mse_change=delta/queries,
            gain_percent=-100*delta/max(baseline_sum,1e-12),worst_query_harm=worst,
            failed_groups=len(bad),failed_contexts=len({g.rsplit('|',1)[0] for g in bad}),
            attempted_reads=0 if t>1 else attempts)
    rows.append(row(1.00001));i=0
    while i<len(events):
        score=events[i]['score'];j=i
        while j<len(events) and events[j]['score']==score:
            e=events[j];d=e['delta'];delta+=d;benefit+=max(0.,-d);harm+=max(0.,d);worst=max(worst,d)
            counts['accepted']+=1;counts['useful']+=d < -1e-9;counts['harmful']+=d > 1e-9
            counts['correct']+=e['correct'];counts['wrong']+=not e['correct'];counts['missing']+=e['missing']
            for g in e['groups']:
                sums[g]+=d
                if sums[g]/group_sizes[g]>1e-9:bad.add(g)
                else:bad.discard(g)
            j+=1
        rows.append(row(score));i=j
    if rows[-1]['threshold']!=0:rows.append(row(0.))
    return rows


def at_threshold(rows,t):
    i=bisect.bisect_right([-r['threshold'] for r in rows],-t)-1
    return dict(rows[max(i,0)],threshold=t,attempted_reads=0 if t>1 else rows[-1]['attempted_reads'])


def summarize(rows,current):
    best=lambda rs:min(rs,key=lambda r:(r['net_mse_change'],r['harm_sum'],-r['threshold']))
    return dict(current=at_threshold(rows,current),strict_group_best=best([r for r in rows if r['failed_groups']==0]),
        empirical_harm_count_frontier={str(k):best([r for r in rows if r['harmful']<=k]) for k in [0,1,2,4,8,16]},
        unconstrained_best=best(rows))
