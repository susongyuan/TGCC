"""Full-policy real-window training with an immutable retrieval encoder.

The disposable audit performs complete batch48 updates, DCP/Adam/RNG restore,
and exact next-update equivalence. Formal training requires that source-bound
audit in warmup and joint phases plus real-data and reference-learning gates.
"""
from pathlib import Path
from datetime import timedelta
import argparse, hashlib, json, os, random, shutil, signal, sys, time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'code/src'),str(ROOT/'code/packages/openpi-client/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ.setdefault('TGR_GEOMETRY_CACHE_PATH','/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3')
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from safetensors.torch import load_file
from tgr.factory import create,ensure_legacy_imports
ensure_legacy_imports()
from tgcc.checkpoint import sources,digest,rng_state
from tgcc.distribution_data import successful_anchor_order
from tgcc.distribution_joint import effective_distribution_loss
from tgcc.full_distribution_data import make_loader
from tgcc.unrolled_distribution_data import UnrolledCorpusData,UnrolledBatchPreparer
from tgcc.transition_features import FEATURE_SCHEMA,EXECUTION_TARGET
from tgcc.optimization import clip_subsystems
from tgcc.corrected_readiness import validate_corrected_readiness
from tgcc.full_distribution_joint import FullDistributionSystem,full_groups,full_rates,full_rates_by_name,required_audit_gradients,full_accumulation,update_forward_groups,set_complete_update_accumulation,memory_parameter,history_valid_count,reference_benefit_ok
from tgcc.unrolled_distribution_data import perturbation_plan
from tgcc.distribution_checkpoint import save_distribution,resume_distribution,load_distribution_student,load_distribution_critic
from tgcc.retrieval_adapter import RetrievalAdapter
from train_distribution import local,clip,state_fingerprint
from train import BASE,PARENT,PARENT_SHA


def gradient_report(model,device):
    from tgcc.joint import policy_parameter_group
    keys=('critic','action','execution','planning','retrieval','retrieval_attention','retrieval_gates','vision','vlm','expert','action_io','graphs','memory','memory_probe','correction','semantic_memory')
    result={k:torch.zeros((),device=device) for k in keys}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        value=local(p.grad).float().square().sum()
        key=('critic' if name.startswith('critic.') else 'retrieval' if name.startswith('retrieval.') else 'memory_probe' if name.startswith('memory_probe.')
             else 'correction' if name.startswith(('correction_query.','correction_head.')) else 'memory' if memory_parameter(name) else policy_parameter_group(name.removeprefix('student.')))
        result[key]+=value
        if name.startswith('retrieval.gates.'):result['retrieval_gates']+=value
        elif name.startswith('retrieval.channels.'):result['retrieval_attention']+=value
        if '.decoder.memory.' in name:result['semantic_memory']+=value
        for head,prefix in (('action','critic.action_density'),('execution','critic.execution_density'),('planning','critic.planning')):
            if name.startswith(prefix):result[head]+=value
    for k,v in result.items():dist.all_reduce(v);result[k]=float(v.sqrt())
    return result


def same_rng(a,b):
    return (a['python']==b['python'] and a['numpy'][0]==b['numpy'][0]
        and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
        and torch.equal(a['torch'],b['torch']) and torch.equal(a['cuda'],b['cuda']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--feature-library',type=Path,required=True)
    p.add_argument('--critic-initialization',type=Path)
    p.add_argument('--student-initialization',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/tgcc/full_distribution.json')
    p.add_argument('--manifest',type=Path,default=ROOT/'reports/hotdog_lazy_seq2_manifest.jsonl')
    p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--audit-phase',choices=('warmup','joint'),default='joint')
    p.add_argument('--audit-start-update',type=int,default=0,help='Disposable audit only: select original seeded windows without pretending to resume trained weights')
    p.add_argument('--readiness',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--critic-regression',type=Path,help='Fixed input-output receipt of the reused critic (required with --critic-initialization)')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    from openpi.models_pytorch import gemma_pytorch as _gemma_pytorch
    attention_implementation=cfg.get('attention_implementation','eager')
    if attention_implementation not in ('eager','sdpa'):raise ValueError('attention_implementation must be eager or sdpa')
    _gemma_pytorch.JOINT_ATTENTION=attention_implementation
    if a.audit_start_update<0 or (a.audit_start_update and not a.audit) or a.audit_start_update+4>cfg['new_updates'] or (a.audit and a.resume is not None):
        raise ValueError('A fresh disposable audit needs a valid original sample offset; formal/resume offsets are forbidden')
    rank,world,gpu=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    accumulation=full_accumulation(world,a.micro_batch,cfg)
    blocks=cfg['memory_unroll_blocks']
    if cfg['new_updates']!=cfg['critic_warmup_updates']+cfg['joint_updates']:
        raise ValueError('Explicit corrected full batch/schedule mismatch')
    device=torch.device('cuda',gpu);torch.cuda.set_device(device);torch.set_num_threads(4)
    fraction=cfg['cuda_memory_fraction']
    if not 0<fraction<=1:raise ValueError('Invalid own-process CUDA memory fraction')
    torch.cuda.set_per_process_memory_fraction(fraction,device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=1800))  # shared-GPU cold starts exceeded 120 s (v3fast)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    source=sources(ROOT);a.output.mkdir(parents=True,exist_ok=True)
    if digest(Path(PARENT)/'model.safetensors')!=PARENT_SHA:raise ValueError('Original TC step5000 changed')
    parent_md=json.loads((Path(PARENT)/'metadata.json').read_text())
    for name,sha in parent_md['implementation_source_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Original implementation changed: '+name)
    data=UnrolledCorpusData(a.manifest,BASE,ROOT/'reports/data_plan_vocabulary.json',digest(a.manifest),blocks,cfg['memory_burnin_blocks'],cfg['cpu_exposure_cache_size'])
    speed_overlay=dict(block_cache=None,retrieval='exact_vectorized_cpu')
    if cfg.get('block_cache_root'):
        from tgcc.speed_cache import BlockCache,data_code_identity
        identity=dict(manifest=digest(a.manifest),plan=data.demonstrations.plan_sha,normalization=data.demonstrations.adapter.normalization_sha256,code=data_code_identity(ROOT))
        data.source.block_cache=BlockCache(cfg['block_cache_root'],identity,int(cfg.get('block_cache_cap_gib',6))*2**30,int(cfg.get('block_cache_keep_episodes',3)),prune=(rank==0))
        speed_overlay['block_cache']=dict(root=str(data.source.block_cache.root),identity=identity,cap_gib=int(cfg.get('block_cache_cap_gib',6)),keep_episodes=int(cfg.get('block_cache_keep_episodes',3)))
    preparer=UnrolledBatchPreparer(a.feature_library,BASE,device);ident=preparer.library.identity
    preparer.perturbation=cfg.get('perturbation') or None;preparer.perturbation_seed=cfg['seed']
    if cfg.get('retrieval_visual_cache'):preparer.visual_cache={};preparer.visual_cache_stats=dict(hits=0,misses=0,calls=0)
    if preparer.perturbation and any(type(cfg.get(k)) not in (int,float) for k in ('memory_probe_weight','correction_weight','memory_lr','correction_lr')):
        raise ValueError('v2.1 recipe requires memory_probe_weight, correction_weight, memory_lr and correction_lr')
    if ident.get('feature_schema')!=FEATURE_SCHEMA or ident.get('execution_target')!=EXECUTION_TARGET:
        raise ValueError('Corrected training requires rebuilt delta support')
    if (digest(a.manifest)!=ident['demonstrations_manifest_sha256'] or data.demonstrations.adapter.normalization_sha256!=ident['normalization_sha256']
        or ident.get('temporary_incomplete') or len(preparer.library.train_indices)!=5888):raise ValueError('Complete source/library identity mismatch')
    if a.critic_initialization is not None and a.critic_regression is None:
        raise ValueError('Critic reuse requires its fixed input-output regression receipt')
    dimensions=dict(context_width=ident['context_width'],after_width=ident['after_width'],planning_classes=ident['planning_classes'],
        hidden=cfg['hidden'],horizon=32,action_width=23,action_mixtures=cfg['action_mixtures'],execution_mixtures=cfg['execution_mixtures'],
        minimum_scale=cfg['minimum_scale'],maximum_scale=cfg['maximum_scale'])
    _,student=create(BASE,'tri_graph','cpu',PARENT)
    policy_initialization_updates=5000;student_initialization_sha256=PARENT_SHA
    if a.student_initialization is not None:
        md=json.loads((a.student_initialization/'metadata.json').read_text())
        if (md.get('full_model_updates')!=5000 or md.get('audit_disposable') is not False or not md.get('training_complete')):
            raise ValueError('Correction requires a verified completed full5000 student initializer')
        student=load_distribution_student(student,a.student_initialization,md)
        policy_initialization_updates=md['policy_cumulative_updates'];student_initialization_sha256=md['student_sha256']
    export_dtypes={n:str(v.dtype) for n,v in student.state_dict().items()}
    adapter=RetrievalAdapter(expert_width=student.base.action_in_proj.out_features,context_width=ident['context_width'],after_width=ident['after_width'],initial_gate=cfg['retrieval_initial_gate'])
    model=FullDistributionSystem(student,adapter,dimensions,cfg)
    critic_receipt=None
    if a.critic_initialization is not None:
        critic_receipt=load_distribution_critic(model.critic,a.critic_initialization,dict(feature_schema=FEATURE_SCHEMA,execution_target=EXECUTION_TARGET,
            distribution_config=dimensions,feature_library_sha256=digest(a.feature_library/'identity.json'),normalization_sha256=ident['normalization_sha256'],
            feature_codec_sha256=ident['feature_codec_sha256'],retrieval_index_sha256=ident['retrieval_index_sha256']),regression=a.critic_regression)
        if critic_receipt['source_critic_actual_total_updates']!=cfg['critic_initialization_actual_updates']:
            raise ValueError('Config critic_initialization_actual_updates must equal the reused critic actual total updates')
    elif cfg.get('critic_initialization_actual_updates',0):
        raise ValueError('Config declares a reused critic but none was given')
    # Fresh heads: execution now predicts actual changes with corrected masks.
    # The frozen retrieval encoder is separate storage and never in the optimizer.
    if any(x is y for x in model.parameters() for y in preparer.encoder.model.parameters()):raise ValueError('Retrieval encoder aliases student')
    metadata=dict(config=cfg,source_sha256=source,parent_checkpoint=PARENT,parent_model_sha256=PARENT_SHA,parent_aliases=parent_md.get('aliases',{}),
        encoder_checkpoint=PARENT,distribution_config=dimensions,retrieval_adapter_config=adapter.record(),policy_export_dtypes=export_dtypes,
        world_size=world,micro_batch=a.micro_batch,grad_accum_steps=accumulation,plan_sha256=digest(data.demonstrations.plan_path),
        feature_library_sha256=digest(a.feature_library/'identity.json'),feature_library=str(a.feature_library.resolve()),
        demonstration_manifest=str(a.manifest.resolve()),policy_cache_identity_sha256=None,audit_disposable=a.audit,
        feature_artifact_sha256=ident['artifact_sha256'],training_sequences=len(data),original_training_sequences=len(data.source.train_indices),training_episodes=184,training_anchors=5888,
        validation_anchors=512,training_unit='real_contiguous_memory_windows',memory_unroll_blocks=blocks,memory_burnin_blocks=cfg['memory_burnin_blocks'],feature_schema=FEATURE_SCHEMA,execution_target=EXECUTION_TARGET,
        parameter_scope='full_student_vision_vlm_expert_action_io_three_graphs_semantics_memory_plus_critic_and_retrieval_adapter',
        frozen_modules=['independent_retrieval_encoder'],fsdp_layout='root_flat_bfloat16_forward_fp32_master_and_reduction',
        critic_initialization_actual_updates=cfg.get('critic_initialization_actual_updates',0),nominal_critic_warmup_budget=cfg['critic_warmup_updates'],fresh_critic_only_updates=0,
        critic_initialization=critic_receipt,auxiliary_heads_deployed=False,
        auxiliary_heads=['memory_probe (past skill history from carried memory)','correction_query+correction_head (actual correction delta from fused perturbed query and reference)'],
        policy_initialization_updates=policy_initialization_updates,student_initialization_sha256=student_initialization_sha256,audit_phase=a.audit_phase if a.audit else None,
        optimizer_reset_reason='verified_student_weights_only; fresh_delta_critic_and_reference_adapter; fresh_optimizer',
        labels='curated successful demonstrations; no independent terminal receipts for every source episode',
        failed_or_synthetic_targets_used=False,retrieval_per_actual_exposure=True)
    for k in ('feature_codec_sha256','retrieval_index_sha256','demonstrations_manifest_sha256','normalization_sha256','encoder_model_sha256'):metadata[k]=ident[k]
    metadata['speed_overlay']=speed_overlay
    metadata['recovery_signal']=dict(perturbation=cfg.get('perturbation'),perturbed_query='same perturbed state through the frozen codec state slice',
        memory_probe_weight=cfg.get('memory_probe_weight'),correction_weight=cfg.get('correction_weight'),memory_lr=cfg.get('memory_lr'),correction_lr=cfg.get('correction_lr'),
        reference_gate_init=cfg['retrieval_initial_gate'],hinge_focus='effectively_perturbed_blocks',failure_data_used=False,target_actions_unchanged=True)
    metadata['attention_implementation']=attention_implementation
    metadata['audit_start_update']=a.audit_start_update if a.audit else None
    metadata['audit_offset_is_sample_selection_not_prior_updates']=a.audit
    metadata['fsdp_gradient_accumulation']='complete_update_fp32_unsharded_local_grad; first_allgather_last_reduce_scatter; no_synthetic_windows'
    if not a.audit:
        if a.readiness is None:raise ValueError('Full recipe requires current readiness')
        gate=json.loads(a.readiness.read_text())
        validate_corrected_readiness(gate,metadata)
    model.to(torch.float32).to(device)
    mesh=init_device_mesh('cuda',(world,))
    fully_shard(model,mesh=mesh,reshard_after_forward=False,mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=False))
    opt=torch.optim.AdamW(full_groups(model,cfg),betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],foreach=False,fused=False)
    torch.manual_seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);random.seed(cfg['seed']+rank)
    step,cursor=resume_distribution(a.resume,model,opt,metadata,rank) if a.resume else (0,0)
    if a.audit:cursor=a.audit_start_update*cfg['effective_batch']
    order=successful_anchor_order(data.rows,cfg['new_updates']*cfg['effective_batch'],cfg['seed'])
    def loader_at(position):return make_loader(data,order,position,rank,world,a.micro_batch,a.workers,cfg['seed'])
    loader,iterator=loader_at(cursor);consumed=cursor
    if rank==0:(a.output/'run_identity.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def update(position,need_gradients=False):
        nonlocal consumed
        model.train();model.critic_only=(a.audit_phase=='warmup' if a.audit else position<cfg['critic_warmup_updates'])
        opt.zero_grad(set_to_none=True)
        rate_position=position+cfg['critic_warmup_updates'] if a.audit and a.audit_phase=='joint' else position
        rates=full_rates_by_name(rate_position,cfg)
        for group in opt.param_groups:group['lr']=rates[group['name']]
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        batches=[next(iterator) for _ in range(accumulation)]
        n=len(batches)*a.micro_batch
        known=sum(int(s['next_masks'].sum()) for batch in batches for s in batch)
        skills=sum(int(item['labels']['next_mask'].sum()) for batch in batches for s in batch for item in s['policy']['sequence'])
        history_valid=sum(history_valid_count(s) for batch in batches for s in batch)
        perturbed_total=sum(int(perturbation_plan(cfg['seed'],s['manifest_index'],blocks,preparer.perturbation)[0].sum()) for batch in batches for s in batch)
        counts=torch.tensor([n*blocks,n*blocks,known,skills,history_valid,perturbed_total],device=device,dtype=torch.long);dist.all_reduce(counts)
        global_counts={k:int(v) for k,v in zip(('action','execution','planning'),counts[:3])}
        losses=[];manifest_indices=[];component_sums={};reference_sums={};reference_count=0
        forward_groups=update_forward_groups(batches)
        for group_index,(group,scale) in enumerate(forward_groups):
            # The only FSDP communication is the first root all-gather and
            # final gradient reduce-scatter of this complete batch48 update.
            # Rank-local groups may differ in count without crossing collectives.
            set_complete_update_accumulation(model,group_index==len(forward_groups)-1)
            batch=preparer(group);manifest_indices.extend(batch['manifest_indices'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                parts,valid=model(batch)
                loss=(effective_distribution_loss(parts,valid,global_counts,world,scale,cfg['loss_weights'])+cfg['skill_loss_weight']*parts['skill_sum']*world/max(int(counts[3]),1)
                      +cfg.get('memory_probe_weight',0)*parts['memory_history_sum']*world/max(int(counts[4]),1)
                      +cfg.get('correction_weight',0)*parts['correction_sum']*world/max(int(counts[5]),1))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite complete full loss')
            for name in ('action','execution','planning'):
                value=parts[name].detach()*valid[name]*world/max(global_counts[name],1)
                component_sums[name]=component_sums.get(name,0)+value
            for name in ('flow','reference_improvement'):
                if name in parts:component_sums[name]=component_sums.get(name,0)+parts[name].detach()/scale
            component_sums['skill']=component_sums.get('skill',0)+parts['skill_sum'].detach()*world/max(int(counts[3]),1)
            component_sums['memory_history']=component_sums.get('memory_history',0)+parts['memory_history_sum'].detach()*world/max(int(counts[4]),1)
            component_sums['correction']=component_sums.get('correction',0)+parts['correction_sum'].detach()*world/max(int(counts[5]),1)
            component_sums['perturbed_fraction']=component_sums.get('perturbed_fraction',0)+batch['perturbed_blocks'].float().sum()/(n*blocks)
            component_sums['effective_fraction']=component_sums.get('effective_fraction',0)+batch['perturbation_effective'].float().sum()/(n*blocks)
            component_sums['selfcheck_skipped_fraction']=component_sums.get('selfcheck_skipped_fraction',0)+batch['perturbation_skipped'].float().sum()/(n*blocks)
            reference_count+=valid['reference_conditioned']
            for name,value in valid.get('reference_metrics',{}).items():
                reference_sums[name]=reference_sums.get(name,0)+value/scale
            loss.backward();losses.append(float(loss.detach()))
            consumed+=world*len(group)
        measurements={**{'loss_'+k:v for k,v in component_sums.items()},**reference_sums}
        for name,value in measurements.items():
            dist.all_reduce(value);measurements[name]=float(value/world)
        conditioned=torch.tensor(reference_count,device=device);dist.all_reduce(conditioned)
        gradients=gradient_report(model,device) if need_gradients else None
        norm,subsystem_norms=clip_subsystems(model,device,cfg['grad_clip']);opt.step();torch.cuda.synchronize()
        values=torch.tensor([sum(losses),time.monotonic()-started],device=device,dtype=torch.float64)
        loss_sum=values[:1].clone();dist.all_reduce(loss_sum)
        seconds=values[1:].clone();dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
        return dict(new_update=position+1,full_model_updates=(0 if model.critic_only else position+1 if a.audit else max(0,position+1-cfg['critic_warmup_updates'])),phase='critic_warmup' if model.critic_only else 'full_joint',memory_unroll_blocks=blocks,reference_conditioned=int(conditioned),measurements=measurements,gradient_subsystems=subsystem_norms,loss=float(loss_sum[0]/world),seconds=float(seconds[0]),
            gradient_norm=norm,gradients=gradients,global_valid_counts=counts.tolist(),data_cursor=consumed,
            local_manifest_indices=manifest_indices if a.audit else None,
            local_forward_group_count=len(forward_groups),
            local_forward_group_sizes=[len(group) for group,_ in forward_groups],
            local_forward_group_burnin=[len(group[0]['policy_burnin']['sequence']) if group[0].get('policy_burnin') else 0 for group,_ in forward_groups],
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    def save(position,export=False):
        # Check free disk before materializing a second atomic full checkpoint.
        required=sum(p.numel()*p.element_size() for p in model.parameters())*3+8*2**30
        previous=a.output/'checkpoints/latest'
        if cfg.get('checkpoint_storage')=='native_zstd' and previous.exists():
            prior=json.loads((previous/'metadata.json').read_text())
            # Only a joint checkpoint has complete Adam state. A warmup
            # checkpoint must never underestimate the first joint save.
            if prior.get('full_model_updates',0)>0 and prior.get('source_sha256')==source:
                footprint=sum(p.stat().st_size for p in previous.rglob('*') if p.is_file())
                required=int(footprint*1.25)+8*2**30
        if export:required+=8*2**30
        enough=torch.tensor(int(shutil.disk_usage(a.output).free>required),device=device)
        dist.all_reduce(enough,op=dist.ReduceOp.MIN)
        if not enough:raise RuntimeError('Insufficient own-run disk headroom for atomic full checkpoint; existing checkpoint preserved')
        if sources(ROOT)!=source:raise ValueError('Frozen full source changed')
        return save_distribution(a.output/('audit_checkpoint' if a.audit else 'checkpoints'),model,opt,position,metadata,rank,export=export)
    def reference_validation():
        from tgcc.data import move_tree
        from tgcc.retrieval_adapter import RetrievalContext
        model.eval();records=[];lib=preparer.library
        def context(record):
            tensor=lambda value,dtype:torch.as_tensor(np.asarray(value),device=device,dtype=dtype)[None]
            return RetrievalContext(tensor(preparer.normalizer.transform(record['actual_native']),torch.float32),
                tensor(record['action_mask'],torch.bool),tensor(record['context'],torch.float32),
                tensor(record['after_features'],torch.float32),torch.ones(1,device=device,dtype=torch.bool),
                tensor(record['after_feature_mask'],torch.bool))
        with torch.no_grad():
            for i in lib.validation_indices[::32]:
                anchor=lib[int(i)];sample=data.demonstrations[int(anchor['manifest_index'])]
                if sample['episode_id']!=anchor['episode_id']:raise ValueError('Heldout source changed')
                ref=lib.query(anchor['context'],exclude_episode_id=anchor['episode_id'])[0][0]
                replacement=lib[next(int(j) for j in lib.train_indices if lib.rows[int(j)]['episode_id']!=ref['episode_id'])]
                if ref['split']!='train' or replacement['split']!='train':raise ValueError('Heldout reference leakage')
                probe=dict(item=move_tree(sample['sequence'][0],device),reference=context(ref),replacement=context(replacement))
                with torch.autocast('cuda',dtype=torch.bfloat16):measure=model(dict(audit_reference_probe=probe))
                row={k:float(v) for k,v in measure.items()}
                if any(not np.isfinite(v) for v in row.values()):raise ValueError('Nonfinite heldout reference probe')
                records.append(dict(episode_id=anchor['episode_id'],**row))
        if len({r['episode_id'] for r in records})!=16:raise ValueError('Heldout reference coverage incomplete')
        if (max(r['reference_effect_rms'] for r in records)<=1e-9
                or max(r['replacement_effect_rms'] for r in records)<=1e-9):
            raise ValueError('Actual policy is insensitive to reference contents')
        return dict(status='passed',scope='16_heldout_episodes_same_input_noise_time_target_per_pair',
            task_success_or_recovery_gain_proven=False,records=records)
    if a.audit:
        records=[]
        for position in range(a.audit_start_update,a.audit_start_update+3):
            report=update(position,True)
            required=required_audit_gradients(a.audit_phase,tuple(report['gradients']),cfg)
            if any(report['gradients'][k]<=0 or not np.isfinite(report['gradients'][k]) for k in required):
                raise AssertionError('Missing full/data/head gradient: '+str(report))
            if a.audit_phase=='warmup' and any(v!=0 for k,v in report['gradients'].items() if k not in required):
                raise AssertionError('Warmup updated a policy/reference module')
            records.append(report)
            if rank==0:print(json.dumps(dict(stage='full_disposable_update',**report)),flush=True)
        isolated={}
        if a.audit_phase=='joint':
            # Production-configuration (FSDP) evidence that each auxiliary loss reaches its intended path:
            # one real micro-batch, one forward, backward on that loss alone, grouped gradient norms, no step.
            for part_name in ('correction_sum','memory_history_sum'):
                model.train();model.critic_only=False;opt.zero_grad(set_to_none=True)
                probe_loader,probe_iterator=loader_at(consumed);probe_samples=next(probe_iterator)
                probe_group,_=update_forward_groups([probe_samples])[0]
                set_complete_update_accumulation(model,True)
                probe_batch=preparer(probe_group)
                with torch.autocast('cuda',dtype=torch.bfloat16):probe_parts,_=model(probe_batch)
                probe_parts[part_name].float().sum().backward()
                isolated[part_name]=gradient_report(model,device);opt.zero_grad(set_to_none=True)
                probe_iterator=None;probe_loader=None
            if rank==0:print(json.dumps(dict(stage='isolated_gradients',**{k:{g:round(v,6) for g,v in r.items()} for k,r in isolated.items()})),flush=True)
        metadata['audit_actual_checkpoint_optimizer_updates']=3
        metadata['audit_sample_cursor_update']=a.audit_start_update+3
        checkpoint=save(a.audit_start_update+3)
        fingerprint=state_fingerprint(model,opt);saved_rng=rng_state();saved_cursor=consumed
        # Compare the genuinely uninterrupted next update with a restored one.
        uninterrupted=update(a.audit_start_update+3,True);next_fingerprint=state_fingerprint(model,opt);next_rng=rng_state()
        restored_step,restored_cursor=resume_distribution(checkpoint,model,opt,metadata,rank)
        if (restored_step,restored_cursor)!=(a.audit_start_update+3,saved_cursor) or state_fingerprint(model,opt)!=fingerprint or not same_rng(saved_rng,rng_state()):
            raise AssertionError('Full saved model/Adam/RNG/cursor failed exact roundtrip')
        iterator=None;loader=None;loader,iterator=loader_at(restored_cursor);consumed=restored_cursor
        restarted=update(a.audit_start_update+3,True)
        if (next_fingerprint!=state_fingerprint(model,opt) or not same_rng(next_rng,rng_state())
            or uninterrupted['local_manifest_indices']!=restarted['local_manifest_indices'] or uninterrupted['loss']!=restarted['loss']):
            raise AssertionError('Full next update differs after save/restore')
        reference_check=reference_validation() if a.audit_phase=='joint' else None
        result=dict(status='passed',source_sha256=source,config=cfg,world_size=world,rank=rank,reference_validation=reference_check,
            audit_phase=a.audit_phase,audit_start_update=a.audit_start_update,
            audit_actual_optimizer_steps=5,audit_weights_freshly_initialized=True,
            micro_batch=a.micro_batch,student_initialization_sha256=student_initialization_sha256,
            memory_unroll_blocks=blocks,memory_burnin_blocks=cfg['memory_burnin_blocks'],
            feature_library_sha256=metadata['feature_library_sha256'],actual_next_complete_update_exact=True,
            actual_model_adam_rng_roundtrip=True,formal_optimizer_updates=0,audits=records+[uninterrupted,restarted],isolated_gradients=isolated)
        (a.output/f'full_audit_rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        if rank==0:print(json.dumps(dict(stage='full_audit_passed',**result)),flush=True)
    else:
        stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stopped.__setitem__(0,True))
        while step<cfg['new_updates']:
            report=update(step,step==0 or (step+1)%250==0);step+=1;cursor+=cfg['effective_batch']
            if consumed!=cursor:raise AssertionError('Full sampler drift')
            if step==1 or step%10==0:
                if rank==0:
                    with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
                    print(json.dumps(report),flush=True)
            stop=torch.tensor(int(stopped[0]),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if step%cfg['checkpoint_every']==0 or step==cfg['new_updates'] or bool(stop):
                save(step,export=step==cfg['new_updates'])
                if rank==0:
                    root=a.output/'checkpoints';old=sorted(root.glob('step[0-9][0-9][0-9][0-9][0-9]'))
                    for path in old[:-1]:
                        md=json.loads((path/'metadata.json').read_text())
                        if md['source_sha256']!=source or md['config']!=cfg:raise ValueError('Foreign checkpoint pruning refused')
                        shutil.rmtree(path)
                dist.barrier()
            if bool(stop):break
        if rank==0 and step==cfg['new_updates']:(a.output/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(full_model_updates=step-cfg['critic_warmup_updates'],critic_actual_total_updates=cfg.get('critic_initialization_actual_updates',0)+step,fresh_critic_only_updates=cfg['critic_warmup_updates'],source_sha256=source)))
    iterator=None;loader=None;data.demonstrations.close();dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
