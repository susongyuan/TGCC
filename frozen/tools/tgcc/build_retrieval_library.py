#!/usr/bin/env python3
"""Hash-bound, phase-balanced normal-demonstration support library; no new rollouts."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'code/src'),str(ROOT/'code/packages/openpi-client/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import numpy as np
import torch
from tgcc.retrieval_data import (FrozenVisualEncoder,RetrievalDataset,digest,past_action_summary,
    raw_context_vector,raw_after_vector,after_feature_mask,production_graph_summary)
from tgcc.robot import ActionNormalizer
from tgcc.transition_features import (FEATURE_SCHEMA,EXECUTION_TARGET,
    current_outcome_from_context,outcome_mask,transition_target,fit_transition_statistics,normalize_transition)
from tgr.factory import ensure_legacy_imports
ensure_legacy_imports()
from tgr.data import LazySequenceDataset
from tgr.build_batches import named_proprio,phase_labels,skill_target
PARENT_SHA='80436905542b0ce75d77d17cf908b29846f5497b58189177d406daebdd3e1239'
BASE='/media/data5/ssy/pi05-behavior/checkpoints_pytorch/pi05-b1kpt50-cs32'
PARENT='/media/data5/ssy/TGR/experiments/tc_esgr_v2_approx_full_5ep_20261024_02/selected_step5000_user_stop'


def log(output,value):
    value={'time':time.time(),**value};print(json.dumps(value),flush=True)
    with (output/'build_log.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')


def validate_performance_resume(previous,current,evidence):
    """Allow an audited builder-only change without relaxing source ownership."""
    if previous['manifest_indices']!=current['manifest_indices'] or previous['per_episode']!=current['per_episode']:
        raise ValueError('Resume anchor selection changed')
    old=dict(previous['identity']);new=dict(current['identity'])
    old_sources=dict(old.pop('extraction_source_sha256'));new_sources=dict(new.pop('extraction_source_sha256'))
    old_builder=old_sources.pop('tools/tgcc/build_retrieval_library.py')
    new_builder=new_sources.pop('tools/tgcc/build_retrieval_library.py')
    if old!=new or old_sources!=new_sources:
        raise ValueError('Performance resume changed data or feature sources')
    if (evidence.get('stage')!='preprocess_equivalence' or evidence.get('samples',0)<8 or
            not evidence.get('all_nonvisual_fields_bitwise_equal') or evidence.get('mismatches') or
            not evidence.get('gpu_forward_implementation_unchanged')):
        raise ValueError('Performance resume needs successful actual-sample equivalence')
    if (evidence.get('old_builder_sha256')!=old_builder or
            evidence.get('new_builder_sha256')!=new_builder or
            evidence.get('feature_source_sha256')!=old_sources):
        raise ValueError('Performance equivalence is not bound to both frozen builders')
    return dict(original_extraction_source_sha256=previous['identity']['extraction_source_sha256'],
                performance_resume_verified=True)


def load_episode_cache(cache,cache_rows,indices,rows):
    """A smoke/partial/corrupt episode cannot be silently admitted on resume."""
    with np.load(cache,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    records=json.loads(cache_rows.read_text())
    if [r['manifest_index'] for r in records]!=indices or any(len(v)!=len(indices) for v in arrays.values()):
        raise ValueError('Resume episode cache anchor count/order changed')
    for record,index in zip(records,indices):
        source=rows[index]
        for key in ('episode_id','split','scene_id','sequence_steps','source_sha256','plan_sha256'):
            if record[key]!=source[key]:raise ValueError('Resume episode source identity changed: '+key)
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite resume cache')
    return arrays,records


def select_anchors(rows,per_episode):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['episode_id']].append(i)
    selected=[];coverage={}
    for ep,indices in sorted(groups.items()):
        row=rows[indices[0]];stem=f'episode_{int(ep):08d}'
        annotation=json.loads((Path(row['dataset_root'])/'annotations/task-0045'/f'{stem}.json').read_text())
        phases=annotation['skill_annotation'];chosen=set();missing=[]
        for phase in phases:
            lo,hi=phase['frame_duration']
            candidates=[i for i in indices if lo<=rows[i]['step_id']<hi]
            if not candidates:missing.append(phase['skill_idx']);continue
            middle=(lo+hi)/2;chosen.add(min(candidates,key=lambda i:abs(rows[i]['step_id']-middle)))
        if len(chosen)>per_episode:raise ValueError(f'{ep} has more phases than the anchor budget')
        # Add chronological coverage after guaranteeing each available phase.
        for j in np.linspace(0,len(indices)-1,per_episode,dtype=int):
            chosen.add(indices[j])
            if len(chosen)>=per_episode:break
        if len(chosen)<per_episode:
            for i in indices:
                chosen.add(i)
                if len(chosen)>=per_episode:break
        selected.extend(sorted(chosen,key=lambda i:rows[i]['step_id']))
        coverage[ep]=dict(anchors=len(chosen),available_manifest_anchors=len(indices),
            annotated_phases=len(phases),phases_without_aligned_manifest_anchor=missing,
            selected_phases=sorted({skill_target(annotation,rows[i]['step_id'])['skill_idx'] for i in chosen}))
    return selected,coverage


def transform_state(adapter,normalizer,raw_state):
    from tgr.input_adapter import legacy_state_from_named
    from openpi.policies.b1k_policy import extract_state_from_proprio
    named=named_proprio(raw_state)
    state=extract_state_from_proprio(legacy_state_from_named(named)).astype(np.float32)
    result=np.zeros(32,np.float32);result[:23]=normalizer.transform(state,'state')
    return result


_CPU_DATA=None
_CPU_NORMALIZER=None
_CPU_PLAN=None
_CPU_CACHE_READS=None

def _cpu_initialize(manifest,geometry_path):
    global _CPU_DATA,_CPU_NORMALIZER,_CPU_PLAN,_CPU_CACHE_READS
    torch.set_num_threads(1)
    from tgr import geometry_cache
    expected=geometry_cache.implementation_metadata(root='/media/data5/ssy/TGR')
    if any(digest(ROOT/path)!=sha for path,sha in expected['source_sha256'].items()):
        raise ValueError('CPU preprocessing source/cache identity changed')
    if not geometry_cache.configure(geometry_path,expected):raise ValueError('CPU worker geometry cache rejected')
    os.environ['TGR_GEOMETRY_CACHE_PATH']=str(geometry_path)
    _CPU_CACHE_READS={'lookups':0,'hits':0};original=geometry_cache._reader.lookup
    def measured(key):
        value=original(key);_CPU_CACHE_READS['lookups']+=1;_CPU_CACHE_READS['hits']+=value is not None;return value
    geometry_cache._reader.lookup=measured
    _CPU_DATA=LazySequenceDataset(Path(manifest),BASE,ROOT/'reports/data_plan_vocabulary.json')
    _CPU_NORMALIZER=ActionNormalizer(BASE);_CPU_PLAN=json.loads(_CPU_DATA.plan_path.read_text())


def _cpu_prepare_episode(indices):
    data=_CPU_DATA;normalizer=_CPU_NORMALIZER;plan=_CPU_PLAN
    started=time.monotonic();prepared=[];video_hashes=None;old_reads=dict(_CPU_CACHE_READS)
    for index in indices:
        sample=data[index];row=data.rows[index];frame=row['step_id'];demo=data.current_data;numeric=demo.records.columns
        if video_hashes is None:video_hashes={camera:digest(handle['source_path']) for camera,handle in data.current_reader.handles.items()}
        before,after=sample['sequence'];bn=named_proprio(numeric['observation.state'][frame]);an=named_proprio(numeric['observation.state'][frame+32])
        state=before['observation']['state'][0].numpy();after_state=after['observation']['state'][0].numpy()
        past,pvalid=past_action_summary(numeric['action'],frame,normalizer)
        hindices=list(reversed(range(frame-32,max(-1,frame-17*32),-32)))
        history_states=[transform_state(data.adapter,normalizer,numeric['observation.state'][j]) for j in hindices]
        ht=[float(numeric['timestamp'][j]) for j in hindices];now=float(numeric['timestamp'][frame]);nexttime=float(numeric['timestamp'][frame+32])
        if abs(nexttime-now-32/30)>2e-4:raise ValueError('CPU source prefix time mismatch')
        cam=numeric['observation.cam_rel_poses'][frame];acam=numeric['observation.cam_rel_poses'][frame+32]
        empty_visual=np.zeros(3456,np.float32)
        rawcontext=raw_context_vector(empty_visual,state,past,pvalid,named_proprio=bn,cam_rel_poses=cam,
            conditions=before['conditions'],history_states=history_states,timestamp=now,history_timestamps=ht)
        rawafter=raw_after_vector(empty_visual,after_state,named_proprio=an,cam_rel_poses=acam,conditions=after['conditions'])
        mask=after_feature_mask(an,after['conditions'],visual_width=3456,cam_rel_poses=acam)
        native=np.asarray(numeric['action'][frame:frame+32],np.float32);actions=before['actions'][0,:,:23].numpy()
        if not np.allclose(actions,normalizer.transform(native),atol=2e-6,rtol=2e-6):raise ValueError('CPU action normalization mismatch')
        labels=phase_labels(plan,demo.annotation,demo.meta,frame)
        values=dict(raw_context=rawcontext,raw_after_features=rawafter,raw_after_feature_mask=mask,actions=actions,
            action_mask=np.ones(32,bool),actual_native=native,next_index=np.asarray(int(labels['next'][0]),np.int64),
            next_mask=np.asarray(bool(labels['next_mask'][0])),reference_state=state,reference_after_state=after_state)
        record=dict(manifest_index=index,episode_id=row['episode_id'],split=row['split'],scene_id=row['scene_id'],sequence_steps=row['sequence_steps'],
            before_timestamp=now,after_timestamp=nexttime,source_sha256=row['source_sha256'],video_sha256=video_hashes,
            plan_sha256=row['plan_sha256'],rgb_exposures=[item['provenance']['rgb_exposures'] for item in sample['sequence']],
            phase_supervision_only=bool(labels['next_mask'][0]),independent_episode_id=row['episode_id'])
        # Return only NumPy CPU arrays, not shared-storage Torch tensors or CUDA
        # state. The sole GPU parent executes the unchanged six-image forward.
        observations=[]
        for item in sample['sequence']:
            obs=item['observation'];observations.append({name:{key:value.cpu().numpy().copy() for key,value in obs[name].items()}
                for name in ('image','image_mask')})
        prepared.append((values,record,observations))
    reads={key:_CPU_CACHE_READS[key]-old_reads[key] for key in old_reads}
    return prepared,time.monotonic()-started,reads


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest',type=Path,default=Path('/media/data5/ssy/TGR/reports/hotdog_lazy_seq2_manifest.jsonl'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--per-episode',type=int,default=32)
    p.add_argument('--pca-width',type=int,default=256);p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-samples',type=int,default=4)
    p.add_argument('--cpu-workers',type=int,default=0)
    p.add_argument('--preprocess-equivalence',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-performance-evidence',type=Path)
    p.add_argument('--geometry-cache',type=Path,default=Path('/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3'))
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists() and not a.resume:raise FileExistsError('Use a unique evidence output directory')
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    from tgr import geometry_cache
    expected_geometry=geometry_cache.implementation_metadata(root='/media/data5/ssy/TGR')
    if any(digest(ROOT/path)!=sha for path,sha in expected_geometry['source_sha256'].items()):
        raise ValueError('Existing geometry cache does not match frozen extraction sources')
    if not geometry_cache.configure(a.geometry_cache,expected_geometry):
        raise ValueError('Geometry cache identity or completeness rejected')
    os.environ['TGR_GEOMETRY_CACHE_PATH']=str(a.geometry_cache)
    geometry_reads={'lookups':0,'hits':0}
    original_lookup=geometry_cache._reader.lookup
    def tracked_lookup(key):
        result=original_lookup(key);geometry_reads['lookups']+=1;geometry_reads['hits']+=result is not None
        return result
    geometry_cache._reader.lookup=tracked_lookup
    rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x]
    data=LazySequenceDataset(a.manifest,BASE,ROOT/'reports/data_plan_vocabulary.json')
    normalizer=ActionNormalizer(BASE);plan=json.loads(data.plan_path.read_text())
    selection,coverage=select_anchors(rows,a.per_episode)
    identity=dict(schema='tgcc.normal_support_library.v3',feature_schema=FEATURE_SCHEMA,
        execution_target=EXECUTION_TARGET,demonstrations_manifest_sha256=digest(a.manifest),
        normalization_sha256=data.adapter.normalization_sha256,parent_model_sha256=PARENT_SHA,
        encoder_model_sha256=PARENT_SHA,encoder_frozen=True,retrieval_excludes_query_episode=True,
        raw_train_sequences=sum(r['split']=='train' for r in rows),raw_validation_sequences=sum(r['split']=='validation' for r in rows),
        source_train_episodes=len({r['episode_id'] for r in rows if r['split']=='train'}),source_validation_episodes=len({r['episode_id'] for r in rows if r['split']=='validation'}),
        source_scenes=len({r['scene_id'] for r in rows}),subset_per_episode=a.per_episode,
        graph_source='oracle_upper_bound',robot_world_pose_source='privileged_actual256',
        privileged_graphs_authorized=True,depth_source='absent_in_selected_RGB_demonstrations',
        current_phase_annotation_as_input=False,unauthenticated_held_labels='UNKNOWN',
        unauthenticated_open_cooked_toggled_labels='UNKNOWN',
        failed_or_synthetic_targets_used=False,actual_prefix_time_alignment_checked=True,future_targets_as_context_inputs=False,
        terminal_task_success_verified=False,source_semantics='official_curated_human_demonstration_normal_support',
        geometry_cache_sha256=digest(a.geometry_cache),geometry_cache_implementation_digest=expected_geometry['implementation_digest'],
        planning_classes=len(plan['candidates']),plan_sha256=digest(data.plan_path),proprio_layout_sha256=digest(ROOT/'reports/data_proprio_layout.json'),
        action_layout_sha256=digest(ROOT/'reports/data_action_layout.json'),
        extraction_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'src/tgcc/retrieval_data.py',Path(__file__),ROOT/'src/tgr/build_batches.py',ROOT/'src/tgr/input_adapter.py',ROOT/'src/tgr/runtime_bridge.py']})
    plan_path=a.output/'sampling_plan.json'
    sampling=dict(manifest_indices=selection,per_episode=coverage,identity=identity)
    if plan_path.exists() and json.loads(plan_path.read_text())!=sampling:
        if not a.resume or not a.resume_performance_evidence:
            raise ValueError('Resume sampling/source identity mismatch')
        identity.update(validate_performance_resume(json.loads(plan_path.read_text()),sampling,
            json.loads(a.resume_performance_evidence.read_text())))
        identity['performance_resume_evidence_sha256']=digest(a.resume_performance_evidence)
    if not plan_path.exists():plan_path.write_text(json.dumps(sampling,indent=2)+'\n')
    if a.preprocess_equivalence:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        chosen=selection[:8]
        with ProcessPoolExecutor(max_workers=max(1,a.cpu_workers),mp_context=multiprocessing.get_context('spawn'),
            initializer=_cpu_initialize,initargs=(str(a.manifest),str(a.geometry_cache))) as pool:
            prepared,wall,reads=pool.submit(_cpu_prepare_episode,chosen).result()
        with np.load(a.preprocess_equivalence,allow_pickle=False) as f:reference={k:f[k] for k in f.files}
        reference_rows=json.loads(a.preprocess_equivalence.with_suffix('.rows.json').read_text())
        mismatches=[]
        for j,(values,record,obs) in enumerate(prepared):
            at=next(k for k,r in enumerate(reference_rows) if r['manifest_index']==record['manifest_index'])
            if record!=reference_rows[at]:mismatches.append((j,'row_provenance'))
            for key,value in values.items():
                want=reference[key][at]
                if key in ('raw_context','raw_after_features'):value=value[3456:];want=want[3456:]
                if not np.array_equal(value,want):mismatches.append((j,key))
        report=dict(stage='preprocess_equivalence',samples=len(prepared),cpu_seconds=wall,geometry_reads=reads,
                    mismatches=mismatches,all_nonvisual_fields_bitwise_equal=not mismatches,
                    gpu_forward_implementation_unchanged=True,
                    old_builder_sha256=json.loads((a.preprocess_equivalence.parent/'sampling_plan.json').read_text())['identity']['extraction_source_sha256']['tools/tgcc/build_retrieval_library.py'],
                    new_builder_sha256=digest(Path(__file__)),
                    feature_source_sha256={k:v for k,v in identity['extraction_source_sha256'].items() if k!='tools/tgcc/build_retrieval_library.py'},
                    reference_cache_sha256=digest(a.preprocess_equivalence),
                    reference_rows_sha256=digest(a.preprocess_equivalence.with_suffix('.rows.json')))
        (a.output/'equivalence_report.json').write_text(json.dumps(report,indent=2)+'\n');log(a.output,report)
        if mismatches:raise ValueError('Parallel preprocessing changed support records')
        return
    log(a.output,dict(stage='loading_frozen_vision_only' ,pid=os.getpid(),samples=len(selection),smoke=a.smoke))
    encoder=FrozenVisualEncoder.from_parent(Path(PARENT)/'model.safetensors',PARENT_SHA,a.device)
    vision_identity=encoder.save(a.output) if not (a.output/'vision_encoder.safetensors').exists() else dict(
        vision_encoder_sha256=digest(a.output/'vision_encoder.safetensors'),vision_config_sha256=digest(a.output/'vision_config.json'))
    identity.update(vision_identity)
    log(a.output,dict(stage='vision_loaded',elapsed=time.monotonic()-started,gpu_peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    if a.smoke:selection=selection[:a.smoke_samples]
    groups=defaultdict(list)
    for index in selection:groups[rows[index]['episode_id']].append(index)
    all_arrays=defaultdict(list);all_records=[];timings=[]
    executor=None;futures={};ordered=list(sorted(groups.items()));prefetch=max(1,2*a.cpu_workers)
    if a.cpu_workers:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        executor=ProcessPoolExecutor(max_workers=a.cpu_workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=_cpu_initialize,initargs=(str(a.manifest),str(a.geometry_cache)))
        for pos,(ep,indices) in enumerate(ordered[:prefetch]):
            if not (a.resume and (a.output/f'raw_episode_{ep}.npz').exists()):futures[ep]=executor.submit(_cpu_prepare_episode,indices)
    for position,(ep,indices) in enumerate(ordered):
        nextpos=position+prefetch
        if executor and nextpos<len(ordered):
            nep,nindices=ordered[nextpos]
            if not (a.resume and (a.output/f'raw_episode_{nep}.npz').exists()):futures[nep]=executor.submit(_cpu_prepare_episode,nindices)
        cache=a.output/f'raw_episode_{ep}.npz';cache_rows=a.output/f'raw_episode_{ep}.rows.json'
        if a.resume and cache.exists() and cache_rows.exists():
            arrays,records=load_episode_cache(cache,cache_rows,indices,rows);elapsed=0.
        elif executor:
            epstart=time.monotonic();prepared,cpu_seconds,reads=futures.pop(ep).result()
            arrays=defaultdict(list);records=[]
            for values,record,observations in prepared:
                tick=time.monotonic();visual=encoder(observations)
                values['raw_context'][:3456]=visual[0];values['raw_after_features'][:3456]=visual[1]
                values['raw_visual']=visual[0];values['raw_after_visual']=visual[1]
                for key,value in values.items():arrays[key].append(value)
                records.append(record);timings.append(time.monotonic()-tick)
            for key in geometry_reads:geometry_reads[key]+=reads[key]
            arrays={k:np.stack(v) for k,v in arrays.items()};elapsed=time.monotonic()-epstart
            np.savez_compressed(cache,**arrays);cache_rows.write_text(json.dumps(records)+'\n')
            log(a.output,dict(stage='parallel_preprocessing_episode',episode=ep,cpu_seconds=cpu_seconds,parent_seconds=elapsed,workers=a.cpu_workers))
        else:
            arrays=defaultdict(list);records=[];epstart=time.monotonic();video_hashes=None
            for index in indices:
                tick=time.monotonic();sample=data[index];row=rows[index];frame=row['step_id']
                demo=data.current_data;numeric=demo.records.columns
                if video_hashes is None:
                    video_hashes={camera:digest(handle['source_path']) for camera,handle in data.current_reader.handles.items()}
                observations=[s['observation'] for s in sample['sequence']]
                visual=encoder(observations)
                before,after=sample['sequence'];bn=named_proprio(numeric['observation.state'][frame]);an=named_proprio(numeric['observation.state'][frame+32])
                state=before['observation']['state'][0].numpy();after_state=after['observation']['state'][0].numpy()
                past,pvalid=past_action_summary(numeric['action'],frame,normalizer)
                hindices=list(reversed(range(frame-32,max(-1,frame-17*32),-32)))
                history_states=[transform_state(data.adapter,normalizer,numeric['observation.state'][j]) for j in hindices]
                ht=[float(numeric['timestamp'][j]) for j in hindices]
                now=float(numeric['timestamp'][frame]);nexttime=float(numeric['timestamp'][frame+32])
                if abs(nexttime-now-32/30)>2e-4:raise ValueError('Actual source prefix is not time aligned')
                cam=numeric['observation.cam_rel_poses'][frame];acam=numeric['observation.cam_rel_poses'][frame+32]
                rawcontext=raw_context_vector(visual[0],state,past,pvalid,named_proprio=bn,cam_rel_poses=cam,
                    conditions=before['conditions'],history_states=history_states,timestamp=now,history_timestamps=ht)
                rawafter=raw_after_vector(visual[1],after_state,named_proprio=an,cam_rel_poses=acam,conditions=after['conditions'])
                mask=after_feature_mask(an,after['conditions'],visual_width=visual.shape[1],cam_rel_poses=acam)
                native=np.asarray(numeric['action'][frame:frame+32],np.float32)
                actions=before['actions'][0,:,:23].numpy()
                if not np.allclose(actions,normalizer.transform(native),atol=2e-6,rtol=2e-6):raise ValueError('Action normalization differs from official sample')
                labels=phase_labels(plan,demo.annotation,demo.meta,frame)
                values=dict(raw_context=rawcontext,raw_after_features=rawafter,raw_after_feature_mask=mask,
                    actions=actions,action_mask=np.ones(32,bool),actual_native=native,
                    next_index=np.asarray(int(labels['next'][0]),np.int64),next_mask=np.asarray(bool(labels['next_mask'][0])),
                    reference_state=state,reference_after_state=after_state,raw_visual=visual[0],raw_after_visual=visual[1])
                for key,value in values.items():arrays[key].append(value)
                record=dict(manifest_index=index,episode_id=ep,split=row['split'],scene_id=row['scene_id'],sequence_steps=row['sequence_steps'],
                    before_timestamp=now,after_timestamp=nexttime,source_sha256=row['source_sha256'],video_sha256=video_hashes,
                    plan_sha256=row['plan_sha256'],rgb_exposures=[s['provenance']['rgb_exposures'] for s in sample['sequence']],
                    phase_supervision_only=bool(labels['next_mask'][0]),independent_episode_id=ep)
                records.append(record);timings.append(time.monotonic()-tick)
                if a.smoke:log(a.output,dict(stage='smoke_sample',episode=ep,frame=frame,seconds=timings[-1],raw_visual_width=visual.shape[1],raw_context_width=len(rawcontext),raw_after_width=len(rawafter)))
            arrays={k:np.stack(v) for k,v in arrays.items()};elapsed=time.monotonic()-epstart
            np.savez_compressed(cache,**arrays);cache_rows.write_text(json.dumps(records)+'\n')
        for key,value in arrays.items():all_arrays[key].append(value)
        all_records.extend(records)
        log(a.output,dict(stage='episode_complete',episode=ep,split=rows[indices[0]]['split'],samples=len(records),seconds=elapsed,total_samples=len(all_records),elapsed=time.monotonic()-started))
    if executor:executor.shutdown(wait=True)
    data.close();arrays={k:np.concatenate(v) for k,v in all_arrays.items()}
    if a.smoke:
        report=dict(stage='smoke_passed',samples=len(all_records),elapsed=time.monotonic()-started,
                    median_sample_seconds=float(np.median(timings)),geometry_cache_reads=geometry_reads,raw_visual_width=arrays['raw_visual'].shape[1],
                    expected_context_width=a.pca_width+arrays['raw_context'].shape[1]-arrays['raw_visual'].shape[1],
                    expected_after_width=a.pca_width+arrays['raw_after_features'].shape[1]-arrays['raw_visual'].shape[1],
                    source_identity=identity,formal_library_ready=False)
        (a.output/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n');log(a.output,report);return
    train=np.asarray([r['split']=='train' for r in all_records]);val=~train
    if len({r['episode_id'] for r in all_records if r['split']=='train'})!=184 or len({r['episode_id'] for r in all_records if r['split']=='validation'})!=16:
        raise ValueError('Phase subset must cover all 184/16 source-isolated episodes')
    log(a.output,dict(stage='fitting_train_only_pca',samples=int(train.sum())))
    visual=np.concatenate([arrays['raw_visual'][train],arrays['raw_after_visual'][train]])
    torch.manual_seed(20261027);v=torch.from_numpy(visual);mean=v.mean(0);centered=v-mean
    _,_,components=torch.pca_lowrank(centered,q=a.pca_width,center=False,niter=3)
    components=components.numpy();visual_mean=mean.numpy();width=visual.shape[1]
    projection=lambda x:(x-visual_mean)@components
    context=np.concatenate([projection(arrays['raw_context'][:,:width]),arrays['raw_context'][:,width:]],axis=1)
    after=np.concatenate([projection(arrays['raw_after_features'][:,:width]),arrays['raw_after_features'][:,width:]],axis=1)
    aftermask=np.concatenate([np.ones((len(after),a.pca_width),bool),arrays['raw_after_feature_mask'][:,width:]],axis=1)
    context_mean=context[train].mean(0);context_std=np.maximum(context[train].std(0),.01)
    before=current_outcome_from_context(context,a.pca_width)
    after,aftermask=transition_target(before,after,outcome_mask(before,a.pca_width),aftermask)
    after_mean,after_std,after_supported=fit_transition_statistics(after,aftermask,train)
    after,aftermask=normalize_transition(after,aftermask,dict(after_mean=after_mean,after_std=after_std,after_supported=after_supported))
    codec=dict(visual_mean=visual_mean,visual_components=components,context_mean=context_mean.astype(np.float32),context_std=context_std.astype(np.float32),
               after_mean=after_mean.astype(np.float32),after_std=after_std.astype(np.float32),after_supported=after_supported)
    np.savez(a.output/'codec.npz',**codec)
    final=dict(context=((context-context_mean)/context_std).astype(np.float32),after_features=after,after_feature_mask=aftermask,
        actions=arrays['actions'],action_mask=arrays['action_mask'],actual_native=arrays['actual_native'],next_index=arrays['next_index'],next_mask=arrays['next_mask'],
        reference_state=arrays['reference_state'],reference_after_state=arrays['reference_after_state'],reference_visual_context=projection(arrays['raw_visual']).astype(np.float32))
    final['after_features'][~aftermask]=0
    np.savez_compressed(a.output/'samples.npz',**final)
    with (a.output/'rows.jsonl').open('w') as f:
        for r in all_records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    artifacts={name:digest(a.output/name) for name in ('samples.npz','rows.jsonl','codec.npz')}
    identity.update(context_width=context.shape[1],after_width=after.shape[1],visual_width=a.pca_width,
        train_samples=int(train.sum()),validation_samples=int(val.sum()),index_episodes=sorted({r['episode_id'] for r in all_records if r['split']=='train'}),
        validation_episodes=sorted({r['episode_id'] for r in all_records if r['split']=='validation'}),
        pca_fit_episodes='train_only',pca_after_states_used_only_in_train_basis_fit=True,artifact_sha256=artifacts,
        codec_sha256=artifacts['codec.npz'],feature_codec_sha256=artifacts['codec.npz'],feature_artifact_sha256=artifacts,
        retrieval_index_sha256=hashlib.sha256((artifacts['samples.npz']+artifacts['rows.jsonl']).encode()).hexdigest(),
        valid_training_targets=dict(action=int(train.sum()),execution=int(train.sum()),planning=int(final['next_mask'][train].sum())),
        source_counts_are_anchors_not_independent_contexts=True,
        vector_layout=dict(visual=[0,a.pca_width],robot_state=[a.pca_width,a.pca_width+23],past_actions=[a.pca_width+23,a.pca_width+70],
            side_information=[a.pca_width+70,a.pca_width+211],current_graphs=[a.pca_width+211,a.pca_width+1309],
            graph_history=[a.pca_width+1309,a.pca_width+2794],robot_state_history=[a.pca_width+2794,a.pca_width+2841]),
        retrieval_distance_groups=[dict(name='visual',slices=[[0,a.pca_width]],weight=.20),
            dict(name='current_robot_and_calibration',slices=[[a.pca_width,a.pca_width+23],[a.pca_width+70,a.pca_width+211]],weight=.25),
            dict(name='current_three_graphs',slices=[[a.pca_width+211,a.pca_width+1309]],weight=.30),
            dict(name='three_graph_temporal_history',slices=[[a.pca_width+1309,a.pca_width+2794]],weight=.15),
            dict(name='executed_action_and_robot_history',slices=[[a.pca_width+23,a.pca_width+70],[a.pca_width+2794,context.shape[1]]],weight=.10)])
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    loaded=RetrievalDataset(a.output)
    for i in list(loaded.train_indices[::max(1,len(loaded.train_indices)//40)])+list(loaded.validation_indices[::max(1,len(loaded.validation_indices)//16)]):
        reference,_=loaded.query(loaded.arrays['context'][i],exclude_episode_id=loaded.rows[i]['episode_id'])[0]
        if reference['episode_id']==loaded.rows[i]['episode_id'] or reference['split']!='train':raise ValueError('Retrieval episode leakage')
    report=dict(stage='library_complete',elapsed=time.monotonic()-started,samples=len(loaded),train_samples=len(loaded.train_indices),validation_samples=len(loaded.validation_indices),
                context_width=context.shape[1],after_width=after.shape[1],geometry_cache_reads=geometry_reads,identity_sha256=digest(a.output/'identity.json'),
                median_sample_seconds=float(np.median(timings)) if timings else None,training_ready=False)
    (a.output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n');log(a.output,report)

if __name__=='__main__':main()
