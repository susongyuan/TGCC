"""Dense full-model updates; independent frozen retrieval features stay stable."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .distribution_critic import DistributionCritic,distribution_loss
from .data import move_tree,conditions_from_dict
from .joint import policy_parameter_group


def sequence_time(item):
    values=item['timestamps'] if 'timestamps' in item else [item['timestamp']]
    result=torch.as_tensor(values,device=item['actions'].device,dtype=torch.float64)
    if result.shape!=(len(item['actions']),) or not torch.isfinite(result).all():
        raise ValueError('Real exposure timestamp/batch mismatch')
    return result


def full_accumulation(world,micro,recipe):
    per_rank=recipe['gradient_accumulation_by_world'].get(str(world),0)
    if (world not in (3,4) or world>recipe['max_gpus'] or type(micro) is not int or micro<1
            or per_rank%micro or world*per_rank!=recipe['effective_batch'] or recipe['effective_batch']<1):
        raise ValueError('Full training requires explicit three/four-card batch48 layout; micro must divide the per-rank window count')
    return per_rank//micro


def group_by_burnin(samples):
    """Same-length real burn-in groups, order preserved; memory None and tensor are never mixed."""
    groups={}
    for s in samples:
        n=len(s['policy_burnin']['sequence']) if s.get('policy_burnin') else 0
        groups.setdefault(n,[]).append(s)
    return [groups[k] for k in sorted(groups)]


def update_forward_groups(batches):
    """Real nonempty same-burn-in groups, with original complete-update weights.

    Cardinality may differ by rank. FSDP only communicates at the first
    forward and final backward of this entire sequence, not per group.
    """
    if not batches or any(not samples for samples in batches):
        raise ValueError('A complete accumulation update requires nonempty real batches')
    accumulation=len(batches)
    result=[]
    for samples in batches:
        for group in group_by_burnin(samples):
            result.append((group,accumulation*len(samples)/len(group)))
    return result


def set_complete_update_accumulation(model,last_backward):
    """Keep root unsharded and FP32 local grads until this update's final group.

    The caller must use one root-flat FSDP2 system and must not put any
    explicit collective between the rank-local groups. No synthetic groups
    or repeated supervision are introduced to align the ranks.
    """
    if type(last_backward) is not bool:
        raise ValueError('Explicit complete-update backward position is required')
    model.set_requires_gradient_sync(last_backward)
    model.set_reshard_after_backward(last_backward)
    model.set_is_last_backward(last_backward)


def memory_parameter(name):
    """Student parameters of the carried memory (GRU cell and its output projection); accepts names
    from student.named_parameters() ('decoder.memory...') and from the full system ('student.decoder.memory...')."""
    return name.removeprefix('student.').startswith('decoder.memory')


def memory_width(student):
    module=student.decoder.memory
    width=getattr(module,'hidden_size',None) or getattr(module,'out_features',None)
    if not isinstance(width,int) or width<1:raise ValueError('Cannot infer the carried memory width')
    return width


def history_targets(burnin_sequence,policy_sequence,classes,device):
    """Past known skill history for each supervised block: a multi-hot of next-skill labels that were
    known in earlier blocks (burn-in and earlier window blocks). Validity requires at least one known
    earlier label. Only the past enters; the current block's own label never does."""
    items=list(burnin_sequence or [])+list(policy_sequence);start=len(burnin_sequence or [])
    batch=len(items[0]['labels']['next']);seen=torch.zeros(batch,classes,device=device);any_known=torch.zeros(batch,dtype=torch.bool,device=device)
    targets=[];valid=[]
    for index,item in enumerate(items):
        if index>=start:targets.append(seen.clone());valid.append(any_known.clone())
        label=item['labels']['next'].to(device).long();mask=item['labels']['next_mask'].to(device).bool()
        seen=seen.clone();seen[torch.arange(batch,device=device)[mask],label[mask]]=1.;any_known=any_known|mask
    return torch.stack(targets),torch.stack(valid)


def history_valid_count(sample):
    """Number of supervised blocks of one window with at least one known earlier label (same rule as history_targets)."""
    burnin=sample.get('policy_burnin');items=list(burnin['sequence'] if burnin else [])+list(sample['policy']['sequence']);start=len(burnin['sequence']) if burnin else 0
    known=False;count=0
    for index,item in enumerate(items):
        if index>=start and known:count+=1
        known=known or bool(item['labels']['next_mask'].any())
    return count


def reference_benefit_ok(plain,reference,replacement):
    """Acceptance for an assisted prediction: it must beat both the unassisted and the mismatched-reference errors."""
    return bool(reference<plain and reference<replacement)


def slice_prefix_cache(cache,rows):
    """Row-slice a prefix KV cache (legacy per-layer (k,v) tuples or an HF cache object exposing
    to_legacy_cache/from_legacy_cache). Returns None when the cache type is unknown; the caller then computes all rows."""
    if isinstance(cache,(tuple,list)) and cache and all(isinstance(layer,(tuple,list)) and len(layer)==2 for layer in cache):
        layers=[(k[rows],v[rows]) for k,v in cache]
        return tuple(layers) if isinstance(cache,tuple) else layers
    legacy=getattr(cache,'to_legacy_cache',None);build=getattr(type(cache),'from_legacy_cache',None)
    if callable(legacy) and callable(build):
        return build(tuple((k[rows],v[rows]) for k,v in legacy()))
    return None


FROZEN_POLICY_GROUPS_ALLOWED=('vision','vlm')


def apply_frozen_policy_groups(student,groups):
    """Freeze whole pi0.5 parameter groups (vision tower/projector, PaliGemma VLM) before FSDP wrapping.

    Carried memory (decoder.memory), the graph encoders, the action expert and the action I/O layers stay
    trainable. Frozen groups receive no parameter gradients, no optimizer state and no Adam shards; their
    activations are still differentiated because graph and memory condition tokens enter the VLM prefix.
    Returns the frozen parameter names.
    """
    groups=tuple(groups or ())
    unknown=[g for g in groups if g not in FROZEN_POLICY_GROUPS_ALLOWED]
    if unknown:raise ValueError('Only the vision and vlm policy groups may be frozen: '+','.join(unknown))
    frozen=[]
    for name,p in student.named_parameters():
        if groups and not memory_parameter(name) and policy_parameter_group(name) in groups:
            p.requires_grad_(False);frozen.append(name)
    if groups and not frozen:raise ValueError('frozen_policy_groups matched no parameters')
    return frozen


class FullDistributionSystem(nn.Module):
    def __init__(self,student,retrieval,dimensions,recipe):
        super().__init__();self.student=student;self.student.requires_grad_(True)
        # Training-only auxiliary heads (never used to produce control commands):
        # memory_probe: the carried memory alone predicts the past known skill history;
        # correction head: the fused perturbed query + reference representation predicts the actual correction delta.
        expert_width=retrieval.config['expert_width']
        self.memory_probe=nn.Linear(memory_width(student),dimensions['planning_classes'])
        self.correction_query=nn.Linear(dimensions['context_width'],128)
        self.correction_head=nn.Sequential(nn.Linear(2*expert_width+128,256),nn.GELU(),nn.Linear(256,23))
        self.auxiliary_heads_deployed=False
        self.unused_expert_vocabulary=[]
        for name,p in self.student.named_parameters():
            if 'gemma_expert.lm_head.' in name:
                p.requires_grad_(False);self.unused_expert_vocabulary.append(name)
        self.frozen_policy_parameters=apply_frozen_policy_groups(self.student,recipe.get('frozen_policy_groups',()))
        self.retrieval=retrieval;self.critic=DistributionCritic(dimensions);self.recipe=recipe

    def policy_call(self,function,*args):
        scope=self.recipe.get('activation_checkpointing_scope','all')
        if scope not in ('all','prefix'):raise ValueError('activation_checkpointing_scope must be all or prefix')
        if self.training and torch.is_grad_enabled() and self.recipe.get('activation_checkpointing',False) and (scope=='all' or getattr(function,'__name__','')=='prefix'):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(function,*args,use_reentrant=False)
        return function(*args)

    def forward(self,batch):
        if 'audit_reference_probe' in batch:
            if self.training or torch.is_grad_enabled():raise ValueError('Heldout probe must be evaluation-only')
            return self.reference_probe(batch['audit_reference_probe'])
        from tgr.data import collate_sequences
        from tgr.task_feedback import roll_forward_task_graph
        from openpi.models.model import Observation
        if any(not s.get('policy_imitation_allowed',True) or s.get('source_kind','demonstration')!='demonstration' for s in batch['policy_samples']):
            raise ValueError('Full imitation requires original successful demonstrations')
        length=len(batch['policy_samples'][0]['sequence'])
        if (len(batch['references'])!=length or len(batch['reference_episode_ids'])!=length
                or any(len(s['sequence'])!=length for s in batch['policy_samples'])):
            raise ValueError('Both actual exposures and every unroll block require their own current-context reference')
        expected=self.recipe.get('memory_unroll_blocks',length)
        if length!=expected or length<2:
            raise ValueError('Configured real memory-unroll length mismatch')
        for ids in batch['reference_episode_ids']:
            if len(ids)!=len(batch['query_episode_ids']):
                raise ValueError('Incomplete reference episode provenance')
            if any(str(q)==str(r) for q,r in zip(batch['query_episode_ids'],ids)):
                raise ValueError('Reference own-episode leakage')
        _,parts,counts=distribution_loss(self.critic(batch['density']),batch['density'],self.recipe['loss_weights'])
        if getattr(self,'critic_only',False):
            zero=parts['action']*0
            parts.update(flow=zero,skill_sum=zero,memory_history_sum=zero,correction_sum=zero)
            counts.update(skill_valid=0,flow_sequences=len(batch['policy_samples']),
                          memory_unroll_blocks=length,reference_conditioned=0)
            return parts,counts
        policy=move_tree(collate_sequences(batch['policy_samples']),batch['density']['context'].device)
        if len(policy['sequence'])!=length:raise ValueError('Full causal sequence length mismatch')
        self.student.eval();memory=None;previous=None;previous_time=None;flows=[];skills=[];known=0
        reference_losses=[];reference_count=0;reference_metrics={}
        dropout=self.recipe['retrieval_conditioning_dropout']
        if not 0<=dropout<=1:raise ValueError('Invalid retrieval-conditioning dropout')
        reference_weight=self.recipe['loss_weights'].get('reference_improvement',self.recipe.get('reference_improvement_weight',0))
        if ('reference_improvement_weight' in self.recipe and
                self.recipe['reference_improvement_weight']!=reference_weight):
            raise ValueError('Reference objective enable and loss weight disagree')
        burnin=batch.get('policy_burnin_samples',[])
        burnin_count=0
        if burnin:
            observed=move_tree(collate_sequences(burnin),batch['density']['context'].device)
            with torch.no_grad():
                for item in observed['sequence']:
                    c=conditions_from_dict(item['conditions']);c.memory=memory
                    now=sequence_time(item)
                    if previous is not None:
                        c.task_graph=roll_forward_task_graph(c.task_graph,previous,c.task_candidate_slots,now-previous_time)
                    _,_,_,_,decision=self.student.prefix(Observation.from_dict(item['observation']),c,False)
                    memory=decision['memory'].detach();previous=decision;previous_time=now
                    burnin_count+=1
        classes=self.memory_probe.out_features
        history_t,history_v=history_targets(observed['sequence'] if burnin else None,policy['sequence'],classes,batch['density']['context'].device)
        perturbed=batch.get('perturbed_blocks');effective=batch.get('perturbation_effective');correction_t=batch.get('correction_target');perturbed_ctx=batch.get('perturbed_contexts')
        ablation=batch.get('memory_ablation');histories=[];corrections=[];blocks_n=len(policy['sequence'])
        for k,item in enumerate(policy['sequence']):
            if ablation=='zero' and memory is not None:memory=torch.zeros_like(memory)
            elif ablation=='shuffle' and memory is not None:memory=memory.roll(1,0)
            memory_before=memory if memory is not None else torch.zeros(len(item['actions']),self.memory_probe.in_features,device=item['actions'].device)
            probe=F.binary_cross_entropy_with_logits(self.memory_probe(memory_before.float()),history_t[k],reduction='none').mean(-1)
            histories.append((probe*history_v[k]).sum())
            c=conditions_from_dict(item['conditions']);c.memory=memory
            now=sequence_time(item)
            if previous is not None:
                c.task_graph=roll_forward_task_graph(c.task_graph,previous,c.task_candidate_slots,now-previous_time)
            state,valid,cache,tokens,decision=self.policy_call(self.student.prefix,Observation.from_dict(item['observation']),c,False)
            reference=batch['references'][k]
            if len(reference.valid)!=len(batch['query_episode_ids']):
                raise ValueError('Reference tensor/provenance batch mismatch')
            if self.training and dropout:
                keep=torch.rand(len(reference.valid),device=reference.valid.device)>=dropout
                reference=replace(reference,valid=reference.valid&keep)
            original_tokens=tokens
            tokens=self.retrieval(tokens,reference)
            if perturbed is not None and correction_t is not None and perturbed_ctx is not None:
                pool=lambda v,m:(v.float()*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
                query=self.correction_query(perturbed_ctx.view(len(item['actions']),blocks_n,-1)[:,k].float())
                fused=torch.cat((pool(*tokens[0]),pool(*tokens[1]),query),-1)
                predicted=self.correction_head(fused)
                focus_c=(perturbed[:,k]&effective[:,k]).float()
                corrections.append(((predicted-correction_t[k].float()).square().mean(-1)*focus_c).sum())
            reference_count+=int(reference.valid.sum())
            for channel,metrics in self.retrieval.last_metrics.items():
                for name,value in metrics.items():
                    reference_metrics.setdefault(channel+'.'+name,[]).append(value.detach())
            actions=item['actions'];noise=self.student.base.sample_noise(actions.shape,actions.device)
            time=self.student.base.sample_time(len(actions),actions.device)
            velocity=self.policy_call(self.student.velocity,state,valid,cache,tokens,time[:,None,None]*noise+(1-time[:,None,None])*actions,time)
            target=(noise-actions)[...,:23]
            errors=(velocity[...,:23]-target).square().mean((-1,-2))
            flows.append(errors.mean())
            # Same current observation, prefix, noise, time and positive target.
            # A detached unassisted prediction is a comparison, not a negative
            # demonstration or a counterfactual physical after-state.
            if reference_weight>0:
                fraction=self.recipe.get('reference_improvement_fraction',.05)
                if not 0<=fraction<1:raise ValueError('Invalid paired reference improvement fraction')
                # The paired objective is focused on effectively perturbed blocks, where a clean reference is informative.
                focus=reference.valid if perturbed is None else (reference.valid&perturbed[:,k]&effective[:,k])
                # The detached unassisted comparison is computed only for focused rows: rows outside the focus are
                # multiplied by zero below, so their comparison value never enters the loss (identical result, less compute).
                rows=focus.nonzero(as_tuple=True)[0]
                plain_errors=torch.zeros_like(errors)
                if len(rows):
                    with torch.no_grad():
                        x_t=time[:,None,None]*noise+(1-time[:,None,None])*actions
                        sliced=slice_prefix_cache(cache,rows)
                        if sliced is None:
                            plain=self.student.velocity(state,valid,cache,original_tokens,x_t,time)
                            plain_errors=(plain[...,:23]-target).square().mean((-1,-2))
                        else:
                            plain=self.student.velocity(state[rows],valid[rows],sliced,[(a[rows],m[rows]) for a,m in original_tokens],x_t[rows],time[rows])
                            plain_errors=plain_errors.index_put((rows,),(plain[...,:23]-target[rows]).square().mean((-1,-2)))
                hinge=F.relu(errors-(1-fraction)*plain_errors)
                reference_losses.append((hinge*focus).mean())
            mask=item['labels']['next_mask'].bool()
            ce=F.cross_entropy(decision['next'].float(),item['labels']['next'],reduction='none')
            skills.append((ce*mask).sum());known+=int(mask.sum())
            memory=decision['memory'];previous=decision;previous_time=now
        parts['flow']=torch.stack(flows).mean();parts['skill_sum']=sum(skills)
        parts['memory_history_sum']=sum(histories);parts['correction_sum']=sum(corrections) if corrections else parts['action']*0
        counts['memory_history_valid']=int(history_v.sum());counts['correction_valid']=int((perturbed&effective).sum()) if perturbed is not None else 0
        if reference_losses:parts['reference_improvement']=torch.stack(reference_losses).mean()
        counts['skill_valid']=known;counts['flow_sequences']=len(batch['policy_samples'])
        counts['memory_unroll_blocks']=length;counts['reference_conditioned']=reference_count
        counts['memory_burnin_blocks']=burnin_count
        counts['reference_metrics']={name:torch.stack(values).mean() for name,values in reference_metrics.items()}
        return parts,counts

    def reference_probe(self,batch):
        from openpi.models.model import Observation
        item=batch['item'];c=conditions_from_dict(item['conditions']);c.memory=None
        state,valid,cache,tokens,_=self.student.prefix(Observation.from_dict(item['observation']),c,False)
        actions=item['actions'];noise=self.student.base.sample_noise(actions.shape,actions.device)
        time=self.student.base.sample_time(len(actions),actions.device)
        x=time[:,None,None]*noise+(1-time[:,None,None])*actions;target=(noise-actions)[...,:23]
        assisted=self.retrieval(tokens,batch['reference']);swapped=self.retrieval(tokens,batch['replacement'])
        velocities=[self.student.velocity(state,valid,cache,t,x,time)[...,:23] for t in (tokens,assisted,swapped)]
        errors=[(v-target).square().mean() for v in velocities]
        return dict(plain_flow=errors[0],reference_flow=errors[1],replacement_flow=errors[2],
            reference_effect_rms=(velocities[1]-velocities[0]).square().mean().sqrt(),
            replacement_effect_rms=(velocities[2]-velocities[1]).square().mean().sqrt())


def full_groups(model,recipe):
    groups={name:[] for name in ('vision','vlm','expert','action_io','graphs')}
    memory=list(model.memory_probe.parameters());correction=list(model.correction_query.parameters())+list(model.correction_head.parameters())
    for name,p in model.student.named_parameters():
        if not p.requires_grad:continue
        if memory_parameter(name):memory.append(p)
        else:groups[policy_parameter_group(name)].append(p)
    result=[dict(name='critic',params=list(model.critic.parameters()),lr=recipe['critic_lr'])]
    if 'retrieval_gate_lr' in recipe:
        gates=list(model.retrieval.gates.parameters());gate_ids={id(p) for p in gates}
        result.extend([dict(name='retrieval',params=[p for p in model.retrieval.parameters() if id(p) not in gate_ids],lr=recipe['retrieval_adapter_lr']),
            dict(name='retrieval_gates',params=gates,lr=recipe['retrieval_gate_lr'],weight_decay=0.)])
    else:result.append(dict(name='retrieval',params=list(model.retrieval.parameters()),lr=recipe['retrieval_adapter_lr']))
    result+=[dict(name=name,params=values,lr=recipe[name+'_lr']) for name,values in groups.items() if values]  # frozen groups have no parameters
    if len(memory)<=len(list(model.memory_probe.parameters())):raise ValueError('No carried-memory parameters were routed to the memory group')
    return result+[dict(name='memory',params=memory,lr=recipe['memory_lr']),dict(name='correction',params=correction,lr=recipe['correction_lr'])]


RATE_GROUP_NAMES={'critic_lr':'critic','retrieval_adapter_lr':'retrieval','retrieval_gate_lr':'retrieval_gates','vision_lr':'vision','vlm_lr':'vlm',
    'expert_lr':'expert','action_io_lr':'action_io','graphs_lr':'graphs','memory_lr':'memory','correction_lr':'correction'}


def full_rates_by_name(step,recipe):
    """Learning rates keyed by optimizer group name, so frozen (absent) groups cannot misalign the schedule."""
    keys=['critic_lr','retrieval_adapter_lr']
    if 'retrieval_gate_lr' in recipe:keys.append('retrieval_gate_lr')
    keys.extend(['vision_lr','vlm_lr','expert_lr','action_io_lr','graphs_lr','memory_lr','correction_lr'])
    return {RATE_GROUP_NAMES[k]:rate for k,rate in zip(keys,full_rates(step,recipe))}


def required_audit_gradients(phase,report_keys,recipe):
    """Gradient groups the GPU audit must see as positive: critic heads in warmup; in the joint phase every
    reported group except the recipe's frozen policy groups, which legitimately carry no gradient."""
    if phase=='warmup':return ('critic','action','execution','planning')
    frozen=set(recipe.get('frozen_policy_groups',()))
    return tuple(k for k in report_keys if k not in frozen)


def full_rates(step,recipe):
    import math
    warm=recipe['warmup']
    scale=(step+1)/max(1,warm) if step<warm else recipe['min_lr_ratio']+(1-recipe['min_lr_ratio'])*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,recipe['new_updates']-warm))))
    keys=['critic_lr','retrieval_adapter_lr']
    if 'retrieval_gate_lr' in recipe:keys.append('retrieval_gate_lr')
    keys.extend(['vision_lr','vlm_lr','expert_lr','action_io_lr','graphs_lr','memory_lr','correction_lr'])
    if recipe.get('memory_unroll_blocks'):
        joint=step-recipe['critic_warmup_updates']
        if joint<0:policy_scale=0.
        elif joint<warm:policy_scale=(joint+1)/max(1,warm)
        else:policy_scale=recipe['min_lr_ratio']+(1-recipe['min_lr_ratio'])*.5*(1+math.cos(math.pi*min(1,(joint-warm)/max(1,recipe['joint_updates']-warm))))
        return [recipe[key]*(scale if key=='critic_lr' else policy_scale) for key in keys]
    return [recipe[key]*scale for key in keys]
