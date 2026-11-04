import torch
from .encoders import GraphBatch


def perturb(g, node_drop=0.0, edge_drop=0.0, noise_std=0.0, generator=None):
    x = g.features.clone()
    valid = g.valid.clone()
    rel = g.relations.clone()
    if not 0 <= node_drop <= 1 or not 0 <= edge_drop <= 1 or noise_std < 0:
        raise ValueError("invalid augmentation")
    valid &= (
        torch.rand(valid.shape, device=valid.device, generator=generator) >= node_drop
    )
    rel = rel.masked_fill(
        torch.rand(rel.shape, device=rel.device, generator=generator) < edge_drop, 0
    )
    x = x + torch.randn(x.shape, device=x.device, generator=generator) * noise_std
    return GraphBatch(x, g.semantics, rel, valid, oracle=g.oracle)
