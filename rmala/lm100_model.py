"""Common ~100M trunk; explicit normalized-GLA / V14-LM / full / official HoLA."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import feature
from .lm100_memory import V14Memory

VARIANTS = ('gla', 'v14_full', 'v14_half', 'full', 'hola')


def normalized_gla(q, k, v, g, reconstruction=False):
    from fla.ops.gla import chunk_gla
    d = v.shape[-1]
    width = 1 << d.bit_length()
    augmented = F.pad(v, (0, width-d))
    augmented[..., d] = 1
    args = dict(k=k.to(v.dtype).contiguous(), v=augmented.contiguous(),
                g=g.contiguous(), scale=1., output_final_state=False)
    out, _ = chunk_gla(q=q.to(v.dtype).contiguous(), **args)
    base = out[..., :d].float()/out[..., d:d+1].float().clamp_min(1e-6)
    rec = None
    if reconstruction:
        with torch.no_grad():
            own, _ = chunk_gla(q=k.to(v.dtype).contiguous(), **args)
            rec = own[..., :d].float()/own[..., d:d+1].float().clamp_min(1e-6)
    return base, rec


class GLA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.decay = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.memory = None

    def forward(self, x):
        b,t,d = x.shape
        q0,k0,v = self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        q,k = feature(q0), feature(k0)
        g = F.logsigmoid(self.decay(x).reshape_as(q).float())/16
        base, rec = normalized_gla(q,k,v,g,self.memory is not None)
        if self.memory is not None:
            base = self.memory(q0,k0,v,base,rec)
        return self.proj(base.reshape(b,t,d).to(x.dtype))


def rotary(x):
    t,d = x.shape[1],x.shape[-1]
    pos = torch.arange(t, device=x.device).float()
    inv = torch.exp(torch.arange(0,d,2,device=x.device).float()*(-math.log(10000.)/d))
    angle = pos[:,None]*inv[None,:]
    c,s = angle.cos()[None,:,None,:], angle.sin()[None,:,None,:]
    a,b = x[...,::2].float(),x[...,1::2].float()
    return torch.stack([a*c-b*s,a*s+b*c],-1).flatten(-2).to(x.dtype)


class Full(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim,3*dim,bias=False)
        self.proj = nn.Linear(dim,dim,bias=False)

    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        y=F.scaled_dot_product_attention(rotary(q).transpose(1,2),rotary(k).transpose(1,2),
            v.transpose(1,2),is_causal=True)
        return self.proj(y.transpose(1,2).reshape(b,t,d))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = nn.Linear(dim,2*hidden,bias=False)
        self.down = nn.Linear(hidden,dim,bias=False)

    def forward(self,x):
        a,b = self.up(x).chunk(2,dim=-1)
        return self.down(F.silu(a)*b)


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, variant, idx):
        super().__init__()
        self.n1,self.n2 = nn.RMSNorm(dim,eps=1e-6),nn.RMSNorm(dim,eps=1e-6)
        if variant == 'full':
            self.attn = Full(dim,heads)
        elif variant == 'hola':
            from fla.layers.gated_deltanet import GatedDeltaNet
            self.attn = GatedDeltaNet(hidden_size=dim,expand_v=1,head_dim=128,num_heads=dim//128,
                mode='chunk',use_gate=True,use_short_conv=True,conv_size=4,layer_idx=idx,norm_eps=1e-6,
                use_gdn_swa=True,gdn_swa_evict='betae',gdn_swa_window=64,gdn_swa_chunk=256,
                gdn_swa_gate_init=-4,gdn_swa_cache_norm='rms',gdn_swa_tau_init=1,
                gdn_swa_tau_freeze=True,gdn_swa_cache_kernel='sdpa')
        else:
            self.attn = GLA(dim,heads)
        self.mlp = SwiGLU(dim,hidden)
        self.hola = variant == 'hola'

    def forward(self,x):
        a=self.attn(self.n1(x))
        x=x+(a[0] if self.hola else a)
        return x+self.mlp(self.n2(x))


class LM100(nn.Module):
    def __init__(self, variant, vocab_size=32000, dim=640, layers=16, heads=10,
                 gate_state=None, seed=41001, checkpoint_blocks=False):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant,self.dim,self.layers,self.heads = variant,dim,layers,heads
        self.checkpoint_blocks=checkpoint_blocks
        self.ffn_dim = (1728 if variant=='full' else 1536) if dim==640 else 3*dim
        torch.manual_seed(seed)
        self.embedding=nn.Embedding(vocab_size,dim)
        self.blocks=nn.ModuleList([Block(dim,heads,self.ffn_dim,variant,i) for i in range(layers)])
        self.norm=nn.RMSNorm(dim,eps=1e-6)
        self.head=nn.Linear(dim,vocab_size,bias=False)
        self.head.weight=self.embedding.weight
        # Only initialize our own nn.Linear/Embedding modules. HoLA's specialized
        # decay/cache parameters keep the official constructor initialization.
        self.apply(self._initialize)
        # Attach gates AFTER common initialization so all three GLA trunks match bitwise.
        if variant.startswith('v14'):
            for block in self.blocks:
                block.attn.memory=V14Memory(value_dim=dim//heads,alpha=1. if variant=='v14_full' else .5)
                if gate_state is not None:
                    block.attn.memory.load_v14_gate(gate_state)

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.,std=.02)
            if getattr(module,'bias',None) is not None:
                nn.init.zeros_(module.bias)

    def hidden(self,ids):
        x=self.embedding(ids)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                from torch.utils.checkpoint import checkpoint
                x=checkpoint(block,x,use_reentrant=False)
            else:
                x=block(x)
        return self.norm(x)

    def forward(self,ids):
        return self.head(self.hidden(ids))

    def losses(self, ids, targets):
        # Fused CE still materializes logits, but avoids an FP32 vocabulary copy.
        from fla.modules import FusedCrossEntropyLoss
        logits=self(ids)
        return FusedCrossEntropyLoss(ignore_index=-100,reduction='none')(
            logits.reshape(-1,logits.shape[-1]),targets.reshape(-1)).reshape_as(targets)

    def memory_stats(self):
        return [b.attn.memory.stats for b in self.blocks if isinstance(b.attn,GLA) and b.attn.memory is not None]
