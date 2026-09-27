"""Offline frozen-candidate interventions; never used by the live router."""
import math


def replay(base, candidate, probability, distance, threshold, hamming,
           rate=.05, gate='frozen', populated=True):
    if gate not in ('frozen', 'bypass', 'oracle'):
        raise ValueError(gate)
    if not 0 <= rate <= 1:
        raise ValueError(rate)
    assert len(base)==len(candidate)==len(probability)==len(distance)
    errors=[];mask=[];attempts=0;helpful=harmful=0;benefit=damage=0.
    for i,(b,v,p,d) in enumerate(zip(base,candidate,probability,distance)):
        read=populated and attempts < math.floor(rate*(i+1)+1e-9) and d<=hamming
        attempts+=int(read)
        accept=read and (gate=='bypass' or (gate=='frozen' and p>=threshold)
                        or (gate=='oracle' and v<b))
        errors.append(v if accept else b);mask.append(bool(accept))
        if accept and b-v>1e-9:helpful+=1;benefit+=b-v
        if accept and v-b>1e-9:harmful+=1;damage+=v-b
    return dict(errors=errors,mask=mask,attempted_reads=attempts,
                applied_corrections=sum(mask),helpful=helpful,harmful=harmful,
                benefit_sum=benefit,damage_sum=damage)
