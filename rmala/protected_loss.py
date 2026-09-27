"""Training-only protected-group harm cost; inference receives no group labels."""
import torch
from torch.nn import functional as F


def protected_utility_loss(logits,base_error,candidate_error,protected,multiplier=4.):
    if multiplier<1:raise ValueError('Harm multiplier must be >=1')
    gain=base_error-candidate_error;weights=gain.abs()
    cost=torch.where(protected&(gain<0),weights*multiplier,weights)
    # Keep the original denominator so only protected harmful examples change.
    return (F.binary_cross_entropy_with_logits(logits,(gain>0).float(),reduction='none')*cost).sum()/weights.sum().clamp_min(1e-8)
