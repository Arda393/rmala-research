import torch
from torch import nn
from torch.nn import functional as F
from .bank import ExactBank


def feature(x):
    return F.elu(x.float()) + 1.0


def gla_reference(q, k, v, log_decay):
    """Normalized positive-feature GLA, inclusive causal [B,T,H,D].

    Decay is per key channel; both S and z receive the same decay. This is a
    normalized GLA variant, not the unnormalized default fla GLA layer.
    """
    b, t, h, d = q.shape
    s = q.new_zeros(b, h, d, v.shape[-1], dtype=torch.float32)
    z = q.new_zeros(b, h, d, dtype=torch.float32)
    outputs, recon, denominators = [], [], []
    for i in range(t):
        decay = log_decay[:, i].float().exp()
        ki, qi, vi = k[:, i].float(), q[:, i].float(), v[:, i].float()
        s = s * decay.unsqueeze(-1) + ki.unsqueeze(-1) * vi.unsqueeze(-2)
        z = z * decay + ki
        den = (qi*z).sum(-1).clamp_min(1e-6)
        own = (ki*z).sum(-1).clamp_min(1e-6)
        outputs.append(torch.einsum("bhd,bhdv->bhv", qi, s)/den.unsqueeze(-1))
        recon.append(torch.einsum("bhd,bhdv->bhv", ki, s)/own.unsqueeze(-1))
        denominators.append(den)
    return torch.stack(outputs, 1), torch.stack(recon, 1), torch.stack(denominators, 1)


def gla_fla(q, k, v, log_decay):
    """Two chunk_gla calls yield query read and post-write self reconstruction.

    Appended constant channel tracks z with exactly the same gated recurrence.
    Requires an installed fla supporting [B,T,H,D] and scale=1. No silent fallback.
    """
    if not q.is_cuda:
        raise RuntimeError("fla backend requires CUDA")
    from fla.ops.gla import chunk_gla
    # Keep matrix-product operands in the same dtype under bf16 autocast.
    q, k = q.to(v.dtype), k.to(v.dtype)
    # Pad value dimension to a supported power of two after adding z channel.
    vd = v.shape[-1]
    width = 1 << (vd + 1 - 1).bit_length()
    augmented = F.pad(v, (0, width-vd))
    augmented = augmented + F.one_hot(torch.tensor(vd, device=v.device), width).to(v.dtype)
    args = dict(k=k.contiguous(), v=augmented.contiguous(), g=log_decay.contiguous(),
                scale=1.0, output_final_state=False)
    out, _ = chunk_gla(q=q.contiguous(), **args)
    own, _ = chunk_gla(q=k.contiguous(), **args)
    den = out[..., vd].float().clamp_min(1e-6)
    return (out[..., :vd].float()/den.unsqueeze(-1),
            own[..., :vd].float()/own[..., vd:vd+1].float().clamp_min(1e-6), den)


class ResidualAttention(nn.Module):
    def __init__(self, dim, heads, variant="rmala", tau=.3, capacity=8192,
                 topk=8, hard_threshold=.5, payload="residual", policy="lru",
                 backend="reference", temperature=.1, budget_options=None):
        super().__init__()
        if dim % heads or variant not in {"gla", "rmala", "surprise"}:
            raise ValueError("Invalid dimensions or variant")
        if payload not in {"residual", "raw"} or backend not in {"reference", "fla"}:
            raise ValueError("Invalid payload or backend")
        self.dim, self.heads, self.variant = dim, heads, variant
        self.tau, self.capacity, self.topk = tau, capacity, topk
        self.hard_threshold, self.payload, self.policy = hard_threshold, payload, policy
        self.backend, self.temperature = backend, temperature
        self.budget_options=budget_options
        if budget_options is not None and (variant!="rmala" or backend!="reference"):
            raise ValueError("Budgeted reference attention requires variant=rmala, backend=reference")
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.decay = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        if variant == "rmala":
            self.read_gate = nn.Linear(dim//heads+1, 1)
            if budget_options is not None:
                self.budget_gate=nn.Linear(6,1)
        self.gate_cost=None
        if variant == "surprise":
            # Linear neural memory optimized by explicit local loss gradient.
            self.memory_rate = nn.Parameter(torch.tensor(-3.))
            self.memory_momentum = nn.Parameter(torch.tensor(0.))
            self.memory_decay = nn.Parameter(torch.tensor(-5.))
            self.memory_mix = nn.Parameter(torch.tensor(0.))
        self.stats = {}

    def forward(self, x, hard=None):
        hard = not self.training if hard is None else hard
        b, t, _ = x.shape
        h, d = self.heads, self.dim//self.heads
        q0, k0, v = self.qkv(x).reshape(b,t,3,h,d).unbind(2)
        q, k = feature(q0), feature(k0)
        g = F.logsigmoid(self.decay(x).reshape(b,t,h,d).float()) / 16.0
        if self.budget_options is not None:
            from .budget_attention import forward
            return forward(self,x,q,k,v,g,hard)
        scan = gla_reference if self.backend == "reference" else gla_fla
        base, recon, den = scan(q, k, v, g)
        if self.variant == "gla":
            return self.proj(base.reshape(b,t,-1).to(x.dtype))
        if self.variant == "surprise":
            # Titans-inspired diagnostic baseline, NOT a reproduction of Titans.
            w = q.new_zeros(b,h,d,d)
            momentum = torch.zeros_like(w)
            ys = []
            kn, qn = F.normalize(k,dim=-1), F.normalize(q,dim=-1)
            for i in range(t):
                error = torch.einsum("bhd,bhdv->bhv", kn[:,i], w)-v[:,i].float()
                grad = kn[:,i].unsqueeze(-1)*error.unsqueeze(-2)
                momentum = self.memory_momentum.sigmoid()*momentum+grad
                w = (1-self.memory_decay.sigmoid())*w-self.memory_rate.sigmoid()*momentum
                mem = torch.einsum("bhd,bhdv->bhv", qn[:,i], w)
                ys.append(base[:,i]+self.memory_mix.sigmoid()*mem)
            return self.proj(torch.stack(ys,1).reshape(b,t,-1).to(x.dtype))
        residual = v.float()-recon
        err = residual.norm(dim=-1)/v.float().norm(dim=-1).clamp_min(1e-6)
        # Raw denominator is not calibrated confidence. Learned log feature only.
        gate_input = torch.cat([q, den.log().unsqueeze(-1)], -1)
        alpha = self.read_gate(gate_input.to(self.read_gate.weight.dtype)).squeeze(-1).sigmoid()
        banks = [[ExactBank(self.capacity, self.policy) for _ in range(h)] for _ in range(b)]
        outputs = []
        for ti in range(t):
            rows = []
            for bi in range(b):
                hs = []
                for hi in range(h):
                    bank = banks[bi][hi]
                    if float(err[bi,ti,hi].detach()) > self.tau:
                        payload = residual[bi,ti,hi] if self.payload == "residual" else v[bi,ti,hi].float()
                        bank.write(k[bi,ti,hi], payload)
                    a = alpha[bi,ti,hi]
                    y = base[bi,ti,hi]
                    if bank.size and (not hard or bool(a.detach() >= self.hard_threshold)):
                        values, scores = bank.query(q[bi,ti,hi], self.topk)
                        retrieved = (scores.div(self.temperature).softmax(0).unsqueeze(-1)*values).sum(0)
                        weight = torch.ones_like(a) if hard else a
                        correction = retrieved if self.payload == "residual" else retrieved-y
                        y = y+weight*correction
                    hs.append(y)
                rows.append(torch.stack(hs))
            outputs.append(torch.stack(rows))
        flat = [bank for row in banks for bank in row]
        count = b*t*h
        self.stats = dict(write_rate=sum(z.writes for z in flat)/count,
                          retrieval_rate=sum(z.query_calls for z in flat)/count,
                          comparisons=sum(z.comparisons for z in flat),
                          bank_tensor_bytes=sum(z.tensor_bytes for z in flat),
                          bank_entries=sum(z.size for z in flat),
                          alpha_mean=float(alpha.detach().mean()))
        return self.proj(torch.stack(outputs,1).reshape(b,t,-1).to(x.dtype))
