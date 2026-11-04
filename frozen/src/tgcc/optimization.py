"""Clip independent supervised systems without critic dominating policy scale."""
import torch
import torch.distributed as dist


def clip_subsystems(model,device,maximum=1.):
    names=('critic','retrieval','student');squared=torch.zeros(3,device=device,dtype=torch.float64)
    parameters=[]
    for name,p in model.named_parameters():
        if p.grad is None:continue
        group=0 if name.startswith('critic.') else 1 if name.startswith('retrieval.') else 2
        gradient=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
        squared[group]+=gradient.double().square().sum();parameters.append((gradient,group))
    if dist.is_available() and dist.is_initialized():dist.all_reduce(squared)
    norms=squared.sqrt()
    if not torch.isfinite(norms).all():raise FloatingPointError('Nonfinite subsystem gradient')
    scales=(maximum/(norms+1e-6)).clamp(max=1).float()
    for gradient,group in parameters:gradient.mul_(scales[group])
    return float(norms.square().sum().sqrt()),{key:float(v) for key,v in zip(names,norms)}
