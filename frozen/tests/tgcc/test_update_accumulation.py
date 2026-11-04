"""CPU invariants for unequal real groups and FSDP whole-update positions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

# Run with PYTHONPATH=<new-source>/src under that source's normal pytest setup.
from tgcc.full_distribution_joint import update_forward_groups,set_complete_update_accumulation


def sample(index,length):
    return dict(manifest_index=index,policy_burnin=dict(sequence=list(range(length))) if length else None)


def batches(lengths):
    return [[sample(j+k*6,n) for j,n in enumerate(lengths[k*6:(k+1)*6])] for k in range(2)]


def test_actual_failure_shapes_no_missing_or_duplicate_targets():
    rank_lens=[[4]*12,[4]*11+[0],[4]*11+[1],[4]*11+[2]]
    schedules=[update_forward_groups(batches(x)) for x in rank_lens]
    assert [len(s) for s in schedules]==[2,3,3,3]
    for schedule in schedules:
        ids=[s['manifest_index'] for group,_ in schedule for s in group]
        assert sorted(ids)==list(range(12)) and len(ids)==len(set(ids))
        assert all(len({len(s['policy_burnin']['sequence']) if s.get('policy_burnin') else 0 for s in group})==1 for group,_ in schedule)


def test_mean_loss_and_gradient_equal_all_twelve_real_windows():
    parameter=torch.tensor(2.,requires_grad=True)
    windows=batches([4]*11+[0])
    loss=sum(torch.stack([(parameter*(s['manifest_index']+1))**2 for s in g]).mean()/scale for g,scale in update_forward_groups(windows))
    expected=torch.stack([(parameter*(i+1))**2 for i in range(12)]).mean()
    assert torch.allclose(loss,expected,atol=1e-5,rtol=1e-6)
    a=torch.autograd.grad(loss,parameter,retain_graph=True)[0]
    b=torch.autograd.grad(expected,parameter)[0]
    assert torch.allclose(a,b,atol=1e-5,rtol=1e-6)


class FakeFSDP:
    def __init__(self):self.calls=[]
    def set_requires_gradient_sync(self,value):self.calls.append(('sync',value))
    def set_reshard_after_backward(self,value):self.calls.append(('reshard',value))
    def set_is_last_backward(self,value):self.calls.append(('last',value))


def test_each_rank_first_gather_and_last_reduce_have_same_collective_schedule():
    schedules=[]
    for n in (2,3,5,12):
        model=FakeFSDP();ops=[]
        for i in range(n):
            set_complete_update_accumulation(model,i==n-1)
            if i==0:ops.append('ALLGATHER')
            if model.calls[-3][1]:ops.append('REDUCE_SCATTER')
        assert model.calls[-3:]==[('sync',True),('reshard',True),('last',True)]
        assert all(v is False for _,v in model.calls[:-3])
        schedules.append(ops)
    assert schedules==[['ALLGATHER','REDUCE_SCATTER']]*4


@pytest.mark.parametrize('value',[0,1,None,'false'])
def test_flags_require_explicit_bool(value):
    with pytest.raises(ValueError):set_complete_update_accumulation(FakeFSDP(),value)


@pytest.mark.parametrize('value',[[],[[]]])
def test_no_empty_or_synthetic_group(value):
    with pytest.raises(ValueError):update_forward_groups(value)
