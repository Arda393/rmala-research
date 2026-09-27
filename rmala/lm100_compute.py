"""Explicit FLOP accounting scopes; never equate profiler coverage with total."""
import torch
from torch import nn
from .lm100_memory import V14Memory


def collect_memory_stats(model):
    rows=model.memory_stats()
    tensors=[(i,k,v) for i,row in enumerate(rows) for k,v in row.items() if torch.is_tensor(v)]
    result=[dict(row) for row in rows]
    if tensors:
        values=torch.stack([v.detach() for _,_,v in tensors]).cpu().tolist()
        for (i,k,_),value in zip(tensors,values):
            result[i][k]=value
    return result


def compute_estimate(model, batch, length, memory_stats=(), training=True):
    gate_modules = {id(x) for m in model.modules() if isinstance(m,V14Memory) for x in m.modules()}
    dense_weights = sum(m.weight.numel() for m in model.modules()
                        if isinstance(m,nn.Linear) and id(m) not in gate_modules)
    n, d, layers = batch*length, model.dim, model.layers
    dense_forward = 2*n*dense_weights
    variant = model.variant
    if variant == 'full':
        # Logical causal QK and probability*V multiplies, FMA=2.
        attention_forward = layers*2*batch*d*length*(length+1)
        readout_forward = 0
    elif variant == 'hola':
        # Recurrent DeltaNet update/read and logical cache attention approximation.
        # Official chunk kernels can perform additional hardware work.
        attention_forward = layers*n*(7*d*128 + 4*d*(64+min(256,length)/2))
        readout_forward = 0
    else:
        heads=model.heads
        kd=d//heads
        padded_v=1 << kd.bit_length()
        attention_forward=layers*n*heads*(5*kd*padded_v+2*kd)
        # Detached post-write reconstruction is a second forward scan, no backward.
        readout_forward=attention_forward if variant.startswith('v14') else 0
    retrieval=0
    sketches=0
    for m in memory_stats:
        # Approximate float arithmetic for cosine, dequantization and eight features.
        retrieval += int(m['cosine_pairs'])*4*16 + int(m['reads'])*(5*(d//model.heads)+288)
        sketches += int(m['sketch_pairs'])
    factor=3 if training else 1
    return dict(dense_matmul_flops=factor*dense_forward,
        attention_algorithmic_flops_estimate=factor*attention_forward+readout_forward,
        retrieval_float_ops_estimate=retrieval,
        sketch_pairs=sketches,
        total_algorithmic_flops_estimate=factor*(dense_forward+attention_forward)+readout_forward+retrieval,
        scope='FMA=2; dense matmuls plus logical attention estimate; NOT hardware instruction FLOPs',
        exclusions='optimizer, activation/norm/transcendentals, sort/index/integer work, Triton tiling/padding overhead; gate backward not fully counted')
