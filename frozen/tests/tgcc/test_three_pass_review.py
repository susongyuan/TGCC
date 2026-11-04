"""Regression evidence for failures found in the three-pass source review."""
import copy
import numpy as np
import pytest
import torch
from tgcc.transition_features import normalize_transition,fit_transition_statistics,stable_scoring_context


def test_validation_only_fact_never_changes_calibration_target_space():
    targets=np.array([[1.,0.],[3.,0.],[2.,9999.]],np.float32)
    mask=np.array([[True,False],[True,False],[True,True]])
    mean,std,supported=fit_transition_statistics(targets,mask,[0,1])
    result,valid=normalize_transition(targets,mask,dict(after_mean=mean,after_std=std,after_supported=supported))
    assert not valid[:,1].any() and not result[:,1].any()
    assert valid[:,0].all() and result[2,0]==0
    with pytest.raises(ValueError,match='observed training'):
        fit_transition_statistics(np.array([[float('nan')]]),np.ones((1,1),bool),[0])


def test_cached_task_confidence_matches_fresh_extraction_in_both_graph_times():
    x=np.zeros(3097,np.float32)
    for base in (256+211+366,256+1309+495):
        x[base+15*1]=.8;x[base+222+1]=.5
        x[base+15*9]=1.;x[base+222+9]=.25
        x[base+15*10]=1.;x[base+222+10]=.25
        x[base+364]=.9
    actual=stable_scoring_context(x)
    for base in (256+211+366,256+1309+495):
        assert actual[base+15*9]==actual[base+15*10]==0
        assert actual[base+364]==pytest.approx(.4)
    assert np.array_equal(stable_scoring_context(actual),actual)
    assert x[968]==1  # transform never mutates the original observation


def test_reference_measurements_cover_each_block_and_pair_uses_identical_flow_input(monkeypatch):
    from test_corrected_features_learning import long_batch
    model,batch=long_batch(monkeypatch);calls=[];original=model.student.velocity
    def measured(state,valid,cache,tokens,x,time):
        calls.append((x.detach().clone(),time.detach().clone()))
        return original(state,valid,cache,tokens,x,time)
    model.student.velocity=measured
    parts,counts=model(batch)
    assert len(calls)==16 and counts['reference_conditioned']==16
    for assisted,plain in zip(calls[::2],calls[1::2]):
        assert all(torch.equal(a,b) for a,b in zip(assisted,plain))
    assert set(counts['reference_metrics'])=={channel+'.'+name for channel in ('local','subgoal')
        for name in ('gate','input_rms','delta_rms','relative_rms')}
    assert all(torch.isfinite(v).all() and not v.requires_grad for v in counts['reference_metrics'].values())


def test_incomplete_reference_provenance_and_nonfinite_time_are_rejected(monkeypatch):
    from test_corrected_features_learning import long_batch
    from tgcc.full_distribution_joint import sequence_time
    model,batch=long_batch(monkeypatch);batch['reference_episode_ids'][0]=['only_one']
    with pytest.raises(ValueError,match='Incomplete reference'):model(batch)
    with pytest.raises(ValueError,match='timestamp/batch'):
        sequence_time(dict(actions=torch.zeros(1,32,32),timestamp=float('nan')))


def test_full_reference_dropout_and_objective_config_cannot_silently_disagree(monkeypatch):
    from test_corrected_features_learning import long_batch
    model,batch=long_batch(monkeypatch);model.recipe['retrieval_conditioning_dropout']=1.1
    with pytest.raises(ValueError,match='dropout'):model(batch)
    model.recipe['retrieval_conditioning_dropout']=0
    model.recipe['loss_weights']['reference_improvement']=0
    with pytest.raises(ValueError,match='weight disagree'):model(batch)


def readiness_fixture():
    metadata=dict(source_sha256={'code':'hash'},config={'new_updates':6000},feature_library_sha256='f'*64,
        world_size=3,micro_batch=1,student_initialization_sha256='s'*64,memory_unroll_blocks=8,memory_burnin_blocks=4)
    gate=dict(metadata,status='passed',full_data_gate_passed=True,real_unrolled_data_gate_passed=True,
        reference_condition_validation_passed=True,actual_next_complete_update_exact=True,
        clean_source_review_passes=3,gpu_phase_audits={phase:[dict(metadata,status='passed',rank=rank,
            audit_phase=phase,actual_next_complete_update_exact=True,actual_model_adam_rng_roundtrip=True,
            formal_optimizer_updates=0) for rank in range(3)] for phase in ('warmup','joint')})
    return metadata,gate


@pytest.mark.parametrize('change',['initializer','missing_warmup','stale_rank','two_reviews','unroll','reference'])
def test_formal_admission_requires_all_current_gates(change):
    from tgcc.corrected_readiness import validate_corrected_readiness
    metadata,gate=readiness_fixture();validate_corrected_readiness(gate,metadata)
    if change=='initializer':gate['student_initialization_sha256']='wrong'
    if change=='missing_warmup':del gate['gpu_phase_audits']['warmup']
    if change=='stale_rank':gate['gpu_phase_audits']['joint'][1]['source_sha256']={}
    if change=='two_reviews':gate['clean_source_review_passes']=2
    if change=='unroll':gate['real_unrolled_data_gate_passed']=False
    if change=='reference':gate['reference_condition_validation_passed']=False
    with pytest.raises(ValueError):validate_corrected_readiness(gate,metadata)


@pytest.mark.parametrize('field',['student_initialization_sha256','feature_library_sha256','memory_burnin_blocks','audit_phase'])
def test_changed_initializer_or_burnin_refuses_resume_before_mutation(tmp_path,monkeypatch,field):
    import json
    import tgcc.distribution_checkpoint as api
    from test_distribution_checkpoint import metadata
    expected=metadata(2)
    expected['config'].update(schema='tgcc.corrected_full_success_distribution.recipe.v1',critic_initialization_actual_updates=0)
    expected.update(policy_initialization_updates=10000,feature_schema='delta',execution_target='delta',
        memory_unroll_blocks=8,memory_burnin_blocks=4,audit_phase=None,student_initialization_sha256='a'*64,
        feature_library_sha256='b'*64)
    saved=dict(expected,new_updates=1,full_model_updates=0,fresh_critic_only_updates=1,critic_actual_total_updates=1)
    saved[field]='changed'
    (tmp_path/'metadata.json').write_text(json.dumps(saved))
    monkeypatch.setattr(api,'get_model_state_dict',lambda *a,**k:pytest.fail('Reached mutable DCP loading'))
    with pytest.raises(ValueError,match='identity changed'):
        api.resume_distribution(tmp_path,None,None,expected,0)


def test_planning_feedback_does_not_claim_an_unchanged_skill_was_changed():
    from test_retrieval_runtime import controller,raw,Conditions
    runtime,parent,library=controller(planning=2)
    conditions=Conditions(torch.tensor([[False,True]]))
    runtime.prepare(raw(),conditions)
    runtime.sample_actions('cpu',type('Observation',(),{'state':torch.zeros(1,32)})(),conditions=conditions)
    assert runtime.trace['planning_feedback_triggered'] is True
    assert runtime.trace['actual_subtask_selection_changed'] is False
    assert runtime.trace['original_subtask_selection_changed'] is False


def test_memory_burnin_cannot_overwrite_sampler_window_identity():
    from tgcc.unrolled_distribution_data import UnrolledCorpusData
    class Source:
        rows=[dict(episode_id='e',step_id=k*32,sequence_steps=[k*32,(k+1)*32],
            timestamp=k*32/30,scene_id='s',dataset_root='d',source_sha256={'real':'sha'}) for k in range(12)]
        def __getitem__(self,k):
            sequence=[dict(episode_id='e',timestamp=j*32/30,observation={'frame':j}) for j in (k,k+1)]
            return dict(policy=dict(sequence=sequence),raw_contexts=np.zeros((2,3)),raw_after=np.zeros(3),
                raw_after_mask=np.ones(3,bool),actions=np.zeros((32,23)),next_index=0,next_mask=True,episode_id='e')
    data=UnrolledCorpusData.__new__(UnrolledCorpusData)
    data.source=Source();data.windows=[list(range(4,12))];data.blocks=8;data.burnin_blocks=4
    data.lookup={('e',k*32):k for k in range(12)}
    sample=data[0]
    assert sample['manifest_index']==0 and sample['original_manifest_indices']==list(range(4,12))
    assert [s['observation']['frame'] for s in sample['policy_burnin']['sequence']]==list(range(4))


def test_eight_block_activation_recomputation_preserves_loss_gradients_and_rng(monkeypatch):
    from test_corrected_features_learning import long_batch
    model,batch=long_batch(monkeypatch);rng=torch.get_rng_state()
    def run():
        parts,_=model(batch)
        loss=sum(parts[k] for k in ('action','execution','planning','flow','reference_improvement','skill_sum'))
        loss.backward()
        return {k:v.detach().clone() for k,v in parts.items()}, {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None},torch.get_rng_state()
    plain,gradients,after=run();model.zero_grad(set_to_none=True)
    model.recipe['activation_checkpointing']=True;torch.set_rng_state(rng)
    actual,new_gradients,new_after=run()
    assert torch.equal(after,new_after) and gradients.keys()==new_gradients.keys()
    for k in plain:torch.testing.assert_close(actual[k],plain[k],atol=0,rtol=0)
    for k in gradients:torch.testing.assert_close(new_gradients[k],gradients[k],atol=1e-6,rtol=1e-5)


def test_actual_reference_probe_is_evaluation_only_and_compares_contents(monkeypatch):
    from test_corrected_features_learning import long_batch
    model,batch=long_batch(monkeypatch)
    replacement=copy.deepcopy(batch['references'][0]);replacement.normalized_actions+=2
    probe=dict(audit_reference_probe=dict(item=batch['policy_samples'][0]['sequence'][0],
        reference=batch['references'][0],replacement=replacement))
    with pytest.raises(ValueError,match='evaluation-only'):model(probe)
    model.eval()
    with torch.no_grad():result=model(probe)
    assert all(torch.isfinite(v) for v in result.values())
    assert result['reference_effect_rms']>0 and result['replacement_effect_rms']>0
    assert all(p.grad is None for p in model.parameters())


def test_cpu_exposure_cache_is_bounded_detached_from_consumers_and_episode_owned():
    from tgcc.full_distribution_data import ImmutableExposureCache
    class Source:
        episode_id='one'
        def __init__(self):self.calls=[]
        def __getitem__(self,i):
            self.calls.append(i);return dict(image=torch.full((1,4,4,3),i,dtype=torch.uint8),
                graph=np.array([i,0],np.float32),provenance=dict(episode=self.episode_id,frame=i))
    source=Source();cached=ImmutableExposureCache(source,2)
    first=cached[0];first['image'].fill_(99);first['graph'][0]=99;first['provenance']['episode']='mutated'
    again=cached[0]
    assert again['image'].sum()==0 and again['graph'][0]==0 and again['provenance']['episode']=='one'
    assert source.calls==[0]
    cached[1];cached[2];assert len(cached.values)==2
    cached[0];assert source.calls==[0,1,2,0]
    other=Source();other.episode_id='two';new=ImmutableExposureCache(other,2)
    assert new[0]['provenance']['episode']=='two' and cached[0]['provenance']['episode']=='one'


def test_three_and_four_card_layout_preserves_48_windows_and_admission():
    from tgcc.full_distribution_joint import full_accumulation
    from tgcc.corrected_readiness import validate_corrected_readiness
    cfg=dict(gradient_accumulation_by_world={'3':16,'4':12},max_gpus=4,effective_batch=48)
    assert full_accumulation(3,1,cfg)==16 and full_accumulation(4,1,cfg)==12
    assert full_accumulation(4,2,cfg)==6 and full_accumulation(4,6,cfg)==2
    with pytest.raises(ValueError):full_accumulation(4,5,cfg)
    with pytest.raises(ValueError):full_accumulation(2,1,cfg)
    metadata,gate=readiness_fixture();metadata['world_size']=4;gate['world_size']=4
    for phase,records in gate['gpu_phase_audits'].items():
        for record in records:record['world_size']=4
        records.append(dict(records[0],rank=3))
    validate_corrected_readiness(gate,metadata)
    gate['gpu_phase_audits']['joint'].pop()
    with pytest.raises(ValueError):validate_corrected_readiness(gate,metadata)
