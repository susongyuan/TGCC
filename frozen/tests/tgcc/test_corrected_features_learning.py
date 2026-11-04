import copy
import json
import numpy as np
import pytest
import torch
from tgcc.transition_features import (current_outcome_from_context,outcome_mask,
    transition_target,fit_transition_statistics,stable_scoring_context)
from tgcc.retrieval_data import FrozenFeatureCodec
from tgcc.retrieval_data import production_graph_summary
from tgcc.retrieval_adapter import RetrievalAdapter
from tgcc.unrolled_distribution_data import contiguous_windows
from test_full_distribution_flow import setup


def test_micro1_raw_exposure_timestamp_matches_batched_collation():
    from tgcc.full_distribution_joint import sequence_time
    actual=sequence_time(dict(actions=torch.zeros(1,32,32),timestamp=32/30))
    batched=sequence_time(dict(actions=torch.zeros(2,32,32),timestamps=[32/30,64/30]))
    assert actual.shape==(1,) and actual[0]==batched[0]
    with pytest.raises(ValueError,match='timestamp/batch'):
        sequence_time(dict(actions=torch.zeros(2,32,32),timestamp=32/30))


def test_knownness_does_not_become_execution_error_or_index_feature():
    before=np.zeros(1518,np.float32);after=before.copy()
    # Same scene/physical state, only Oracle goal evidence becomes available.
    after[921]=1
    first=outcome_mask(before);second=outcome_mask(after)
    delta,valid=transition_target(before,after,first,second)
    assert not valid[921] and delta[921]==0
    assert np.isfinite(delta).all()


def test_production_task_summary_is_invariant_to_oracle_goal_confidence():
    def graph(role):
        return dict(features=torch.zeros(1,2,1,32),valid=torch.ones(1,2,1,dtype=torch.bool),
                    roles=torch.full((1,2,1),role,dtype=torch.long),relations=torch.zeros(1,2,1,1,dtype=torch.long))
    conditions=dict(global_graph=graph(1),local_graph=graph(1),task_graph=graph(9))
    before=production_graph_summary(conditions)
    history=production_graph_summary(conditions,historical=True,history_timestamps=[0.],timestamp=1.)
    conditions['task_graph']['features'][...,0]=1.
    assert np.array_equal(before,production_graph_summary(conditions))
    assert np.array_equal(history,production_graph_summary(conditions,historical=True,history_timestamps=[0.],timestamp=1.))


def test_execution_target_measures_change_not_static_instance_offset():
    first=np.zeros(1518,np.float32);first[437]=7.5
    second=first.copy();second[437]+=.2
    valid=np.ones(1518,bool)
    delta,_=transition_target(first,second,valid,valid)
    shifted,_=transition_target(first+3,second+3,valid,valid)
    assert delta[437]==pytest.approx(.2,abs=1e-6)
    assert np.allclose(delta,shifted,atol=1e-6)


def test_equivalent_quaternion_sign_is_not_an_execution_failure():
    first=np.zeros(1518,np.float32);first[307:311]=[0,0,0,1]
    first[346:350]=1  # actual left-EEF quaternion validity, not masked padding
    second=first.copy();second[307:311]*=-1
    delta,_=transition_target(first,second,outcome_mask(first),outcome_mask(second))
    assert outcome_mask(first)[307:311].all()
    assert not delta[307:311].any()
    x=np.zeros(3097,np.float32);x[354:358]=[0,0,0,1]
    changed=x.copy();changed[354:358]*=-1
    assert np.array_equal(stable_scoring_context(x),stable_scoring_context(changed))


def test_unknown_at_either_end_is_not_fabricated_delta():
    first=np.array([np.nan,3],np.float32);second=np.array([4,5],np.float32)
    delta,valid=transition_target(first,second,np.array([False,True]),np.ones(2,bool))
    assert np.array_equal(delta,[0,2]) and np.array_equal(valid,[False,True])


def test_current_outcome_mapping_never_reads_past_or_future():
    x=np.arange(3097,dtype=np.float32)
    actual=current_outcome_from_context(x)
    assert np.array_equal(actual[:279],x[:279])
    assert np.array_equal(actual[279:],x[326:1565]) and len(actual)==1518


def test_statistics_fit_train_only_and_track_unobserved_targets():
    x=np.array([[1,100],[3,200],[99999,99999]],np.float32)
    mask=np.array([[True,False],[True,False],[True,True]])
    mean,std,supported=fit_transition_statistics(x,mask,np.array([0,1]))
    assert np.array_equal(mean,[2,0]) and np.array_equal(supported,[True,False])
    assert std[0]==1


def test_scoring_velocity_filter_does_not_change_native_policy_state():
    state=np.arange(3097,dtype=np.float32);original=state.copy()
    filtered=stable_scoring_context(state)
    assert np.array_equal(state,original)
    assert np.array_equal(filtered[256:259],[0,0,0])
    assert not filtered[329:351].any()
    assert np.array_equal(filtered[259:279],state[259:279])
    retained=np.ones(1098,bool)
    retained[[366+9*15,366+10*15,366+364]]=False
    assert np.array_equal(filtered[467:1565][retained],state[467:1565][retained])
    facts=np.ones(1518,np.float32)
    mask=outcome_mask(facts)
    assert not mask[256:259].any() and not mask[282:304].any()


def test_old_absolute_library_is_rejected_before_any_use(tmp_path):
    (tmp_path/'identity.json').write_text(json.dumps({'context_width':3097}))
    with pytest.raises(ValueError,match='rebuild corrected library'):
        FrozenFeatureCodec.load(tmp_path)


def test_memory_windows_do_not_pad_cross_episode_or_validation():
    rows=[]
    for ep in ['first','second']:
        for k in range(5):
            rows.append(dict(episode_id=ep,step_id=k*32,sequence_steps=[k*32,(k+1)*32],
                timestamp=k*32/30,split='train',scene_id='scene',dataset_root='data',source_sha256={'raw':'sha'}))
    windows=contiguous_windows(rows,4)
    assert len(windows)==4 and all(len(c)==4 for c in windows)
    for chain in windows:assert len({rows[i]['episode_id'] for i in chain})==1
    rows[1]['split']='validation'
    windows=contiguous_windows(rows,4)
    assert all(1 not in chain for chain in windows)
    rows[6]['timestamp']+=.1
    with pytest.raises(ValueError,match='timestamp boundary'):contiguous_windows(rows,4)


def long_batch(monkeypatch):
    model,batch=setup(monkeypatch)
    model.recipe.update(memory_unroll_blocks=8,reference_improvement_weight=.2,
                        reference_improvement_fraction=.05)
    model.retrieval=RetrievalAdapter(16,8,6,initial_gate=.02)
    # The real expert reads both local and subgoal tokens. The original toy
    # parent reads only local tokens; use both for the two-channel gradient test.
    def velocity(state,valid,cache,tokens,x,time):
        condition=sum(value.mean(1) for value,mask in tokens)/len(tokens)
        return model.student.base.action_out_proj(model.student.base.action_in_proj(x)+condition[:,None])
    model.student.velocity=velocity
    base=batch['policy_samples'][0]['sequence'][0]
    sequence=[]
    for k in range(8):
        item=copy.deepcopy(base);item['observation']['exposure']=k
        item['observation']['input'].requires_grad_(True)
        item['timestamps']=[k*32/30]*2;sequence.append(item)
    batch['policy_samples'][0]['sequence']=sequence
    batch['references']=[copy.deepcopy(batch['references'][0]) for _ in range(8)]
    batch['reference_episode_ids']=[['other_first','other_second'] for _ in range(8)]
    return model,batch


def test_eight_block_memory_has_credit_path_to_first_observation(monkeypatch):
    model,batch=long_batch(monkeypatch);memories=[]
    handle=model.student.decoder.memory.register_forward_hook(lambda module,args,out:memories.append(out))
    parts,counts=model(batch)
    assert model.student.seen==list(range(8)) and counts['memory_unroll_blocks']==8
    assert len(memories)==7
    memories[-1].sum().backward()
    first=batch['policy_samples'][0]['sequence'][0]['observation']['input']
    assert first.grad is not None and first.grad.abs().sum()>0
    handle.remove()


def test_real_burnin_is_detached_then_eight_blocks_learn(monkeypatch):
    model,batch=long_batch(monkeypatch)
    policy=copy.deepcopy(batch['policy_samples'][0]);policy['sequence']=policy['sequence'][:4]
    for k,item in enumerate(policy['sequence']):
        item['observation']['exposure']=k-4;item['timestamps']=[k*32/30]*2
    for k,item in enumerate(batch['policy_samples'][0]['sequence']):item['timestamps']=[(k+4)*32/30]*2
    batch['policy_burnin_samples']=[policy]
    parts,counts=model(batch)
    assert model.student.seen==list(range(-4,8))
    assert counts['memory_burnin_blocks']==4
    parts['flow'].backward()
    assert all(item['observation']['input'].grad is None for item in policy['sequence'])
    first=batch['policy_samples'][0]['sequence'][0]['observation']['input']
    assert first.grad is not None and first.grad.abs().sum()>0


def test_paired_reference_objective_reaches_attention_without_extra_noise(monkeypatch):
    model,batch=long_batch(monkeypatch)
    rng=torch.get_rng_state().clone()
    parts,counts=model(batch)
    assert torch.equal(rng,torch.get_rng_state())
    assert 'reference_improvement' in parts and counts['reference_conditioned']==16
    (parts['flow']+.2*parts['reference_improvement']).backward()
    for channel in model.retrieval.channels.values():
        assert channel.in_proj_weight.grad is not None and channel.in_proj_weight.grad.abs().sum()>0
    assert all(float(v['relative_rms'])>0 for v in model.retrieval.last_metrics.values())


def test_critic_warmup_does_not_update_policy_or_reference(monkeypatch):
    model,batch=long_batch(monkeypatch);model.critic_only=True
    parts,counts=model(batch)
    (parts['action']+parts['execution']+.2*parts['planning']).backward()
    assert model.student.seen==[] and counts['reference_conditioned']==0
    assert all(p.grad is None for p in model.student.parameters())
    assert all(p.grad is None for p in model.retrieval.parameters())
    assert any(p.grad is not None for p in model.critic.parameters())


def test_subsystem_clipping_keeps_small_policy_gradient(monkeypatch):
    from tgcc.optimization import clip_subsystems
    model,batch=long_batch(monkeypatch)
    critic=next(model.critic.parameters());student=next(model.student.parameters())
    critic.grad=torch.ones_like(critic)*1000
    student.grad=torch.ones_like(student)*.0001
    expected=student.grad.clone()
    _,norms=clip_subsystems(model,'cpu',1.)
    assert norms['critic']>1 and torch.equal(student.grad,expected)


def test_corrected_controller_retains_native_limit_advisory_extension():
    from tgcc.distribution_runtime import DistributionController
    from tgcc.native_controller_policy import install
    from test_retrieval_runtime import controller,raw,Conditions
    subclass=type('DistributionController',(DistributionController,),{})
    install(subclass)
    runtime,parent,library=controller(action=2)
    runtime.__class__=subclass
    conditions=Conditions(torch.ones(1,2,dtype=torch.bool))
    runtime.prepare(raw(),conditions)
    observed=type('Observation',(),{'state':torch.zeros(1,32)})()
    runtime.sample_actions('cpu',observed,conditions=conditions)
    assert runtime.trace['statistical_feedback_is_task_terminal'] is False
    assert runtime.trace['native_control_policy']=='simulator_native_limits; no_extra_chunk_range_veto'
