"""Per-sequence, per-head memory. Never share a bank across documents/batches.

Routing/eviction is discrete; selected tensors retain their autograd graph during
training. Inference callers should use torch.no_grad(). Exact retrieval costs
O(M*d), even when only top-k payloads are returned.
"""
from abc import ABC, abstractmethod
import torch
from torch.nn import functional as F


class MemoryBank(ABC):
    @abstractmethod
    def write(self, key, value): ...

    @abstractmethod
    def query(self, query, topk): ...

    @abstractmethod
    def evict(self): ...

    @property
    @abstractmethod
    def size(self): ...


class ExactBank(MemoryBank):
    def __init__(self, capacity, policy="lru"):
        if capacity < 0 or policy not in {"lru", "fifo", "magnitude"}:
            raise ValueError("Invalid capacity or eviction policy")
        self.capacity, self.policy = capacity, policy
        self.entries = []
        self.clock = 0
        self.query_calls = self.comparisons = self.writes = 0

    @property
    def size(self):
        return len(self.entries)

    @property
    def tensor_bytes(self):
        return sum((k.numel()*k.element_size() + v.numel()*v.element_size())
                   for k, v, _, _ in self.entries)

    def write(self, key, value):
        if self.capacity == 0:
            return
        self.clock += 1
        self.writes += 1
        # Functional entries avoid autograd version errors from ring-buffer writes.
        self.entries.append((key.clone(), value.clone(), self.clock, self.clock))
        self.evict()

    def evict(self):
        while self.size > self.capacity:
            col = 2 if self.policy == "lru" else 3
            if self.policy == "magnitude":
                scores = [float(e[1].detach().norm()) for e in self.entries]
            else:
                scores = [e[col] for e in self.entries]
            self.entries.pop(min(range(self.size), key=lambda i: scores[i]))

    def _select(self, query, topk):
        keys = torch.stack([e[0] for e in self.entries])
        scores = F.normalize(keys.float(), dim=-1) @ F.normalize(query.float(), dim=-1)
        self.comparisons += self.size
        return scores.topk(min(topk, self.size))

    def query(self, query, topk=8):
        if topk < 1:
            raise ValueError("topk must be positive")
        self.query_calls += 1
        if not self.entries:
            return None, None
        scores, indices = self._select(query, topk)
        ids = indices.detach().cpu().tolist()
        self.clock += 1
        values = []
        for i in ids:
            k, v, _, born = self.entries[i]
            self.entries[i] = (k, v, self.clock, born)
            values.append(v)
        return torch.stack(values), scores


class HNSWBank(ExactBank):
    """Optional inference-only ANN backend with lazy deletion and index rebuilds.

    CPU transfer/index overhead is real and must be benchmarked. Payload and
    key semantics match ExactBank; no asymptotic latency guarantee is claimed.
    """
    def __init__(self, capacity, dim, policy="lru", ef=64):
        super().__init__(capacity, policy)
        import hnswlib
        self.hnswlib, self.dim, self.ef = hnswlib, dim, ef
        self.index = None
        self.dirty = True

    def write(self, key, value):
        if torch.is_grad_enabled() and (key.requires_grad or value.requires_grad):
            raise RuntimeError("HNSW backend is inference-only")
        super().write(key.detach(), value.detach())
        self.dirty = True

    def _select(self, query, topk):
        # Rebuild after mutations: correct reference adapter, not production ANN.
        if self.dirty:
            self.index = self.hnswlib.Index(space="cosine", dim=self.dim)
            self.index.init_index(max_elements=max(1, self.capacity), ef_construction=100, M=16,
                                  random_seed=17)
            keys = torch.stack([e[0] for e in self.entries]).float().cpu().numpy()
            self.index.add_items(keys, list(range(self.size)))
            self.index.set_ef(max(self.ef, topk))
            self.dirty = False
        ids, _ = self.index.knn_query(query.detach().float().cpu().numpy()[None],
                                    k=min(topk, self.size))
        indices = torch.as_tensor(ids[0].astype("int64"), device=query.device)
        keys = torch.stack([self.entries[i][0] for i in ids[0]])
        scores = F.normalize(keys.float(), dim=-1) @ F.normalize(query.float(), dim=-1)
        return scores, indices
