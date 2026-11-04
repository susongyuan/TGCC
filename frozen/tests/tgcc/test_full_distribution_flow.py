"""Full recipe must propagate through both causal policy exposures."""
import copy,sys
from types import SimpleNamespace
import pytest,torch
from torch import nn
from tgcc.full_distribution_joint import FullDistributionSystem
from tgcc.retrieval_adapter import RetrievalAdapter,RetrievalContext


class Parent(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Module();self.base.action_in_proj=nn.Linear(32,16);self.base.action_out_proj=nn.Linear(16,32)
        self.base.paligemma=nn.Linear(2,16);self.global_encoder=nn.Linear(16,16);self.local_encoder=nn.Linear(16,16);self.task_encoder=nn.Linear(16,16)
        self.decoder=nn.Module();self.decoder.memory=nn.Linear(16,16);self.skill=nn.Linear(16,3)
        self.base.sample_noise=lambda shape,device:torch.ones(shape,device=device)
        self.base.sample_time=lambda n,device:torch.full((n,),.5,device=device)
        self.seen=[]
    def prefix(self,obs,c,train):
        self.seen.append(obs['exposure']);h=self.base.paligemma(obs['input'])
        h=self.global_encoder(h)+self.local_encoder(h)+self.task_encoder(h)
        if c.memory is not None:h=h+self.decoder.memory(c.memory)
        tokens=[(h[:,None,:],torch.ones(2,1,dtype=torch.bool))]*2
        return torch.zeros(2,32),torch.ones(2,1,dtype=torch.bool),None,tokens,dict(next=self.skill(h),memory=h)
    def velocity(self,state,valid,cache,tokens,x,time):
        return self.base.action_out_proj(self.base.action_in_proj(x)+tokens[0][0].mean(1)[:,None])


def setup(monkeypatch):
    monkeypatch.setitem(sys.modules,'tgr.data',SimpleNamespace(collate_sequences=lambda samples:samples[0]))
    monkeypatch.setitem(sys.modules,'tgr.task_feedback',SimpleNamespace(roll_forward_task_graph=lambda graph,*args:graph))
    monkeypatch.setitem(sys.modules,'openpi.models.model',SimpleNamespace(Observation=SimpleNamespace(from_dict=lambda obs:obs)))
    import tgcc.full_distribution_joint as module
    monkeypatch.setattr(module,'conditions_from_dict',lambda d:SimpleNamespace(**d))
    torch.manual_seed(42);policy=Parent()
    model=FullDistributionSystem(policy,RetrievalAdapter(expert_width=16,context_width=8,after_width=6),
        dict(context_width=8,after_width=6,planning_classes=3,hidden=16),
        dict(retrieval_conditioning_dropout=0.,loss_weights=dict(action=1.,execution=1.,planning=.2,flow=1.)))
    sequence=[]
    for k in range(2):
        sequence.append(dict(observation=dict(exposure=k,input=torch.randn(2,2)),actions=torch.zeros(2,32,32),timestamps=[float(k)]*2,
            conditions=dict(task_graph=None,task_candidate_slots=None),labels=dict(next=torch.tensor([0,1]),next_mask=torch.ones(2,dtype=torch.bool))))
    reference=RetrievalContext(torch.randn(2,32,23),torch.ones(2,32,dtype=torch.bool),torch.randn(2,8),torch.randn(2,6),
        torch.ones(2,dtype=torch.bool),torch.ones(2,6,dtype=torch.bool))
    batch=dict(density=dict(context=torch.randn(2,8),actions=torch.zeros(2,32,23),action_mask=torch.ones(2,32,dtype=torch.bool),
        after_features=torch.randn(2,6),next_index=torch.tensor([0,1])),references=[reference,copy.deepcopy(reference)],
        policy_samples=[dict(sequence=sequence)],query_episode_ids=['a','b'],reference_episode_ids=[['c','d'],['e','f']])
    return model,batch


def test_full_native_flow_reaches_prefix_graphs_and_cross_chunk_memory(monkeypatch):
    model,batch=setup(monkeypatch);seen=[]
    handle=model.retrieval.register_forward_hook(lambda *args:seen.append(1))
    parts,counts=model(batch);(parts['flow']+.01*parts['skill_sum']+parts['action']+parts['execution']+parts['planning']).backward()
    assert model.student.seen==[0,1] and len(seen)==2 and counts['skill_valid']==4
    for layer in [model.student.base.paligemma,model.student.global_encoder,model.student.local_encoder,model.student.task_encoder,model.student.decoder.memory]:
        assert layer.weight.requires_grad and layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
    handle.remove()


def test_full_both_references_must_exclude_whole_query_episode(monkeypatch):
    model,batch=setup(monkeypatch);batch['reference_episode_ids'][1][0]='a'
    with pytest.raises(ValueError,match='episode leakage'):model(batch)
    batch['references']=batch['references'][:1]
    with pytest.raises(ValueError,match='Both actual exposures'):model(batch)


def test_full_forbids_collected_failure_imitation(monkeypatch):
    model,batch=setup(monkeypatch);batch['policy_samples'][0]['source_kind']='collection'
    with pytest.raises(ValueError,match='original successful'):model(batch)
