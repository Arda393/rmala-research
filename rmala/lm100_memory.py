"""Causal V14-LM adapter with batched, equivalent prefix-credit admission.

V14 was a frozen 16-dimensional synthetic experiment, not a language model.
This explicitly named adaptation keeps historical admission, real int8 raw
values, centered 15-bit sketches, top-1 retrieval, the eight gate features and
5% prefix caps. Contextual q/k channels replace synthetic namespace features.
Training uses a straight-through gate surrogate; forward decisions stay hard.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .utility_gate import UtilityGate


def prefix_accept(candidates):
    """Greedy 1/20 credit policy, with unused credit carried to later tokens."""
    t = candidates.shape[-1]
    cumulative = candidates.to(torch.int32).cumsum(-1, dtype=torch.int32)
    quota = torch.arange(1, t+1, device=candidates.device, dtype=torch.int32)//20
    lost = (cumulative-quota).clamp_min(0).cummax(-1).values
    accepted = cumulative-lost
    return accepted > F.pad(accepted[..., :-1], (1, 0)), accepted


def historical_threshold(error):
    """V14 order statistic of the previous up-to-64 scores; no future values."""
    t = error.shape[-1]
    windows = F.pad(error, (64, 0), value=-float('inf')).unfold(-1, 64, 1)[..., :t, :]
    largest = windows.topk(4, dim=-1).values
    n = torch.arange(t, device=error.device).clamp_max(64)
    # n-1-floor(.95*n): exact integer expression avoids rounding at n=20/40/60.
    rank = (n-1-(95*n)//100).clamp_min(0)
    threshold = largest.gather(-1, rank.view(1,t,1).expand(*error.shape, 1)).squeeze(-1)
    return torch.where(n < 8, .3, threshold.clamp_min(.3))


def signature_batch(x):
    bits = x[..., :15] > x.mean(-1, keepdim=True)
    powers = 1 << torch.arange(15, device=x.device, dtype=torch.int32)
    return (bits.to(torch.int32)*powers).sum(-1, dtype=torch.int32)


def popcount15(x):
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    x = (x + (x >> 4)) & 0x0f0f
    return (x + (x >> 8)) & 0x1f


class V14Memory(nn.Module):
    def __init__(self, value_dim=64, key_dim=16, alpha=1., gate_threshold=.99, hamming=1):
        super().__init__()
        self.gate = UtilityGate()
        self.key_dim, self.value_dim = key_dim, value_dim
        self.alpha, self.gate_threshold, self.hamming = alpha, gate_threshold, hamming
        # Stored FP32 key + int8 value + FP32 scale + 3 int64 metadata + int16 sketch.
        self.entry_bytes = 4*key_dim + value_dim + 4 + 24 + 2
        self.capacity = 2048//self.entry_bytes
        self.stats = {}

    def load_v14_gate(self, state):
        self.gate.load_state_dict(state)

    def forward(self, q_raw, k_raw, values, base, recon, diagnostics=False):
        b, t, heads, d = base.shape
        rows = b*heads
        def flatten(x):
            return x.permute(0, 2, 1, 3).reshape(rows, t, x.shape[-1])
        # Discrete decisions and real int8 memory have no surrogate value gradient.
        with torch.no_grad():
            q = flatten(q_raw[..., :self.key_dim]).float()
            k = flatten(k_raw[..., :self.key_dim]).float()
            v, z, r = (flatten(x).float() for x in (values, base, recon))
            error = (v-r).norm(dim=-1)/v.norm(dim=-1).clamp_min(1e-6)
            writes, counts = prefix_accept(error >= historical_threshold(error))
            wb, wt = writes.nonzero(as_tuple=True)
            slots = counts[wb, wt].long()-1
            max_writes = max(1, t//20)
            key_table = q.new_zeros(rows, max_writes, self.key_dim)
            value_table = torch.zeros(rows, max_writes, d, device=v.device, dtype=torch.int8)
            scales = v.new_zeros(rows, max_writes)
            sketches = torch.zeros(rows, max_writes, device=v.device, dtype=torch.int16)
            selected = v[wb, wt]
            scale = selected.abs().amax(-1).clamp_min(1e-8)/127
            key_table[wb, slots] = k[wb, wt]
            value_table[wb, slots] = (selected/scale[:, None]).round().clamp(-127,127).to(torch.int8)
            scales[wb, slots] = scale
            sketches[wb, slots] = signature_batch(k[wb, wt]).to(torch.int16)
            # UtilityRouter does not touch LRU timestamps. Online adaptation thus
            # evicts oldest insertions, keeping the last `capacity` admitted values.
            bank_ids = counts[..., None].long()-self.capacity+torch.arange(self.capacity, device=v.device)
            valid = bank_ids >= 0
            ids = bank_ids.clamp_min(0)
            batch_ids = torch.arange(rows, device=v.device)[:, None, None]
            sigs = sketches[batch_ids, ids].to(torch.int32)
            distance = popcount15(sigs ^ signature_batch(q)[..., None]).masked_fill(~valid, 32)
            eligible = (distance.amin(-1) <= self.hamming) & (counts > 0)
            reads, read_counts = prefix_accept(eligible)
            rb, rt = reads.nonzero(as_tuple=True)
            ri = ids[rb, rt]
            rv = valid[rb, rt]
            active_keys = key_table[rb[:, None], ri]
            scores = (F.normalize(active_keys, dim=-1)*F.normalize(q[rb, rt], dim=-1)[:, None]).sum(-1)
            scores = scores.masked_fill(~rv, -float('inf'))
            top = scores.topk(2, dim=-1).values
            # V14 uses argmax: ties select the oldest/first active entry.
            # topk's tied indices are not stable and can choose a different value.
            chosen = ri.gather(-1, scores.argmax(-1,keepdim=True)).squeeze(-1)
            candidate = value_table[rb, chosen].float()*scales[rb, chosen, None]
            active_base = z[rb, rt]
            first = (top[:, 0]*1e5).round()/1e5
            second = torch.where(torch.isfinite(top[:, 1]), top[:, 1], -1.)
            gap = ((top[:, 0]-second)*1e5).round()/1e5
            rms = lambda x: x.square().mean(-1).sqrt().log1p()
            prob = scores.softmax(-1)
            entropy = -(prob*prob.clamp_min(1e-9).log()).sum(-1)
            fs = torch.stack([first, gap, rms(active_base), rms(candidate),
                rms(candidate-active_base), F.cosine_similarity(active_base, candidate, dim=-1),
                entropy, counts[rb,rt].clamp_max(self.capacity).float()/self.capacity], -1)
        # Disable bf16 for the transferred gate's calibrated FP32 features/weights.
        with torch.autocast(device_type=base.device.type, enabled=False):
            probability = self.gate(fs).sigmoid()
            hard = ((probability.detach()*1e5).round()/1e5 >= self.gate_threshold).float()
            weight = hard + (probability-probability.detach()) if self.training else hard
        output = flatten(base).clone()
        output[rb, rt] = output[rb, rt] + self.alpha*weight[:, None]*(candidate-output[rb, rt])
        # Tensor counters avoid one host synchronization per layer.
        self.stats = dict(writes=writes.sum(), reads=reads.sum(), accepted=hard.sum(),
            opportunities=rows*t, cosine_pairs=rv.sum(),
            sketch_pairs=valid.sum(), gate_evaluations=rb.numel(),
            bank_capacity=self.capacity, bank_bytes_per_head=self.capacity*self.entry_bytes,
            training_write_cache_bytes=key_table.numel()*4+value_table.numel()+scales.numel()*4+sketches.numel()*2,
            gate_probability_sum=probability.detach().sum())
        if diagnostics:
            self.diagnostics = dict(writes=writes, write_counts=counts, reads=reads,
                read_counts=read_counts, selected_rows=rb, selected_times=rt,
                candidates=candidate, features=fs, accepted=hard, error=error)
        return output.reshape(b, heads, t, d).permute(0,2,1,3)
