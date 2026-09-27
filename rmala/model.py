import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import ResidualAttention


class FullAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, hard=None):
        b,t,d = x.shape
        q,k,v = self.qkv(x).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2),
                                          v.transpose(1,2), is_causal=True)
        return self.proj(y.transpose(1,2).reshape(b,t,d))


class Block(nn.Module):
    def __init__(self, dim, heads, variant, dropout, ffn_dim=None, **kwargs):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = FullAttention(dim,heads) if variant=="full" else ResidualAttention(dim,heads,variant,**kwargs)
        # Compensate GLA's extra dim x dim decay projection with a narrower MLP.
        # Real parameter counts remain recorded; no dummy parameters are added.
        hidden=ffn_dim or (4*dim if variant=="full" else 4*dim-dim//2)
        self.mlp = nn.Sequential(nn.Linear(dim,hidden), nn.GELU(), nn.Linear(hidden,dim))
        self.drop = nn.Dropout(dropout)

    def forward(self,x,hard=None):
        x=x+self.drop(self.attn(self.n1(x),hard=hard))
        return x+self.drop(self.mlp(self.n2(x)))


class LanguageModel(nn.Module):
    def __init__(self, vocab_size=32000, dim=768, heads=12, layers=12,
                 variant="rmala", dropout=.1, **kwargs):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size,dim)
        self.blocks = nn.ModuleList([Block(dim,heads,variant,dropout,**kwargs) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim,vocab_size,bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, ids, hard=None):
        # Shared deterministic positions; untrained length extrapolation is not assumed.
        t=ids.shape[1]
        pos=torch.arange(t,device=ids.device).float().unsqueeze(1)
        inv=torch.exp(torch.arange(0,self.dim,2,device=ids.device).float()*(-math.log(10000.)/self.dim))
        pe=torch.zeros(t,self.dim,device=ids.device)
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*inv),torch.cos(pos*inv)
        x=self.embedding(ids)+pe.to(self.embedding.weight.dtype)
        for block in self.blocks:
            x=block(x,hard=hard)
        return self.head(self.norm(x))

    def memory_stats(self):
        return [b.attn.stats for b in self.blocks if hasattr(b.attn,"stats")]
