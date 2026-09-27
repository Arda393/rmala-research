"""Controlled synthetic token-ID tasks; not the canonical external MQAR suite."""
import torch


def batch(task,batch_size,length,seed,vocab_size=256):
    if task not in {"niah","multi_needle","mqar"} or length<16 or vocab_size<64:
        raise ValueError("Unsupported task/length/vocabulary")
    g=torch.Generator().manual_seed(seed)
    x=torch.randint(vocab_size//2,vocab_size,(batch_size,length),generator=g)
    y=torch.full_like(x,-100)
    pairs=1 if task=="niah" else min(8,length//8)
    for b in range(batch_size):
        keys=torch.randperm(vocab_size//4-4,generator=g)[:pairs]+4
        values=torch.randint(vocab_size//4,vocab_size//2,(pairs,),generator=g)
        # Non-overlapping key-value pairs placed in first half, query in last half.
        slots=torch.randperm(length//4,generator=g)[:pairs]*2
        query_slots=(torch.randperm(length//4,generator=g)[:pairs]*2+2*(length//4)
                     if task=="mqar" else torch.arange(length-2*pairs,length,2))
        for j,(key,value) in enumerate(zip(keys,values)):
            slot=int(slots[j])
            x[b,slot],x[b,slot+1]=key,value
            query=int(query_slots[j])
            x[b,query]=key
            x[b,query+1]=3  # answer placeholder: never ground truth in model input
            y[b,query]=value
    return x,y
