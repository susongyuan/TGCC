"""Complete train corpus, causal raw facts, and a separate frozen query encoder."""
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
from .retrieval_data import (FrozenFeatureCodec,FrozenVisualEncoder,RetrievalDataset,
    raw_context_vector,raw_after_vector,after_feature_mask,past_action_summary)
from .retrieval_adapter import RetrievalContext
from .robot import ActionNormalizer
from .checkpoint import digest,rng_state,restore_rng
from tgr.data import LazySequenceDataset
from tgr.build_batches import named_proprio
from .transition_features import current_outcome_from_context,outcome_mask,transition_target,normalize_transition


class ImmutableExposureCache:
    """Bounded per-episode CPU values, never policy state or changing features.

    Consumers receive independent objects. The source still verifies each
    episode's hashes when opened; a new episode gets a new empty cache.
    """
    def __init__(self,data,capacity):
        self.data=data;self.capacity=capacity;self.values=OrderedDict()
    def __getattr__(self,name):return getattr(object.__getattribute__(self,'data'),name)
    def seed(self,index,value):
        self.values[index]=deepcopy(value);self.values.move_to_end(index)
        while len(self.values)>self.capacity:self.values.popitem(last=False)
    def __getitem__(self,index):
        if index in self.values:
            self.values.move_to_end(index);return deepcopy(self.values[index])
        value=self.data[index];self.seed(index,value);return value


def normalized_state(normalizer,raw):
    from tgr.input_adapter import legacy_state_from_named
    from openpi.policies.b1k_policy import extract_state_from_proprio
    result=np.zeros(32,np.float32)
    result[:23]=normalizer.transform(extract_state_from_proprio(legacy_state_from_named(named_proprio(raw))).astype(np.float32),'state')
    return result


class FullCorpusData:
    def __init__(self,manifest,base,plan,expected_manifest_sha256):
        if digest(manifest)!=expected_manifest_sha256:raise ValueError('Complete demonstration manifest changed')
        self.demonstrations=LazySequenceDataset(manifest,base,plan)
        self.normalizer=ActionNormalizer(base)
        self.exposure_cache_size=0
        self.block_cache=None
        self.rows=self.demonstrations.rows
        self.train_indices=[i for i,r in enumerate(self.rows) if r['split']=='train']
        if len(self.train_indices)!=51714 or len({self.rows[i]['episode_id'] for i in self.train_indices})!=184:
            raise ValueError('Full training must retain all51714 original train units and184 episodes')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        if row['split']!='train':raise ValueError('No validation/test imitation or query targets in training')
        cache=self.block_cache
        if cache is not None:
            value=cache.load(row['episode_id'],index)
            if value is not None:return value
        value=self._compute(index)
        if cache is not None:cache.store(row['episode_id'],index,value)
        return value

    def _compute(self,index):
        row=self.rows[index]
        sample=self.demonstrations[index]
        current=self.demonstrations.current_data
        if self.exposure_cache_size and not isinstance(current,ImmutableExposureCache):
            wrapped=ImmutableExposureCache(current,self.exposure_cache_size)
            for frame,item in zip(row['sequence_steps'],sample['sequence']):
                wrapped.seed(current.anchors.index(frame),item)
            self.demonstrations.current_data=wrapped
        numeric=self.demonstrations.current_data.records.columns
        contexts=[]
        for frame,item in zip(row['sequence_steps'],sample['sequence']):
            named=named_proprio(numeric['observation.state'][frame])
            state=item['observation']['state'][0].numpy()
            history=list(reversed(range(frame-32,max(-1,frame-17*32),-32)))
            history_states=[normalized_state(self.normalizer,numeric['observation.state'][j]) for j in history]
            timestamps=[float(numeric['timestamp'][j]) for j in history]
            now=float(numeric['timestamp'][frame])
            past,valid=past_action_summary(numeric['action'],frame,self.normalizer)
            context=raw_context_vector(np.zeros(3456,np.float32),state,past,valid,
                named_proprio=named,cam_rel_poses=numeric['observation.cam_rel_poses'][frame],
                conditions=item['conditions'],history_states=history_states,timestamp=now,history_timestamps=timestamps)
            if abs(now-float(item['timestamp']))>2e-4 or any(t>now for t in timestamps):
                raise ValueError('Current RGB/state/graph/history is not causally aligned')
            contexts.append(context)
        before,after=sample['sequence'];frame=row['step_id']
        actual=np.asarray(numeric['action'][frame:frame+32],np.float32)
        actions=self.normalizer.transform(actual).astype(np.float32)
        if (actual.shape!=(32,23) or not np.allclose(actions,before['actions'][0,:,:23].numpy(),atol=3e-6,rtol=2e-6)
                or abs(float(after['timestamp'])-float(before['timestamp'])-32/30)>2e-4):
            raise ValueError('Actual native32 prefix, original flow target or after exposure changed')
        named=named_proprio(numeric['observation.state'][frame+32]);cam=numeric['observation.cam_rel_poses'][frame+32]
        after_raw=raw_after_vector(np.zeros(3456,np.float32),after['observation']['state'][0].numpy(),
            named_proprio=named,cam_rel_poses=cam,conditions=after['conditions'])
        mask=after_feature_mask(named,after['conditions'],visual_width=3456,cam_rel_poses=cam)
        return dict(policy=sample,raw_contexts=np.stack(contexts),raw_after=after_raw,
            raw_after_mask=mask,actions=actions,actual_native=actual,
            next_index=int(before['labels']['next'][0]),next_mask=bool(before['labels']['next_mask'][0]),
            episode_id=str(row['episode_id']),manifest_index=int(index),sequence_steps=row['sequence_steps'])


def collate_full(samples):return samples


def make_loader(dataset,order,cursor,rank,world,micro,workers,seed):
    from torch.utils.data import DataLoader
    from .distribution_data import initialize_distribution_worker
    state=rng_state()
    loader=DataLoader(dataset,batch_size=micro,sampler=order[cursor+rank::world],num_workers=workers,
        collate_fn=collate_full,worker_init_fn=initialize_distribution_worker if workers else None,
        generator=torch.Generator().manual_seed(seed+rank),multiprocessing_context='spawn' if workers else None)
    iterator=iter(loader);restore_rng(state)
    return loader,iterator


class FullBatchPreparer:
    """Immutable parent visual features; never reuse changing student features."""
    def __init__(self,library,base,device):
        self.library=RetrievalDataset(library);self.codec=FrozenFeatureCodec.load(library)
        from .speed_cache import install_fast_query;install_fast_query(self.library)
        self.encoder=FrozenVisualEncoder.from_library(library,device)
        self.normalizer=ActionNormalizer(base);self.device=device

    def __call__(self,samples):
        observations=[item['observation'] for sample in samples for item in sample['policy']['sequence']]
        visual=self.encoder(observations).reshape(len(samples),2,3456)
        artifact=self.codec.artifact;contexts=[];afters=[];masks=[]
        for i,sample in enumerate(samples):
            projected=self.codec.visual(visual[i])
            ctx=np.concatenate([projected,sample['raw_contexts'][:,3456:]],axis=-1)
            contexts.append(((ctx-artifact['context_mean'])/artifact['context_std']).astype(np.float32))
            after=np.r_[projected[1],sample['raw_after'][3456:]]
            mask=np.r_[np.ones(projected.shape[-1],bool),sample['raw_after_mask'][3456:]]
            before=current_outcome_from_context(ctx[0],projected.shape[-1])
            after,mask=transition_target(before,after,outcome_mask(before,projected.shape[-1]),mask)
            after,mask=normalize_transition(after,mask,artifact)
            afters.append(after);masks.append(mask)
        contexts=np.stack(contexts)
        def tensor(value,dtype=None):return torch.as_tensor(np.asarray(value),dtype=dtype,device=self.device)
        density=dict(context=tensor(contexts[:,0],torch.float32),actions=tensor([s['actions'] for s in samples],torch.float32),
            action_mask=torch.ones(len(samples),32,dtype=torch.bool,device=self.device),
            after_features=tensor(afters,torch.float32),after_feature_mask=tensor(masks,torch.bool),
            next_index=tensor([s['next_index'] for s in samples],torch.long),
            planning_valid=tensor([s['next_mask'] for s in samples],torch.bool),
            action_valid=torch.ones(len(samples),dtype=torch.bool,device=self.device),
            execution_valid=torch.ones(len(samples),dtype=torch.bool,device=self.device))
        references=[];reference_episodes=[]
        for k in range(2):
            records=[]
            for i,sample in enumerate(samples):
                reference,_=self.library.query(contexts[i,k],exclude_episode_id=sample['episode_id'])[0]
                if reference['split']!='train' or str(reference['episode_id'])==sample['episode_id']:
                    raise ValueError('Full training retrieval leaked its own episode or validation')
                records.append(reference)
            references.append(RetrievalContext(normalized_actions=tensor([self.normalizer.transform(r['actual_native']) for r in records],torch.float32),
                action_mask=tensor([r['action_mask'] for r in records],torch.bool),
                before_features=tensor([r['context'] for r in records],torch.float32),
                after_features=tensor([r['after_features'] for r in records],torch.float32),
                after_feature_mask=tensor([r['after_feature_mask'] for r in records],torch.bool),
                valid=torch.ones(len(samples),dtype=torch.bool,device=self.device)))
            reference_episodes.append([str(r['episode_id']) for r in records])
        return dict(density=density,policy_samples=[s['policy'] for s in samples],references=references,
            query_episode_ids=[s['episode_id'] for s in samples],reference_episode_ids=reference_episodes,
            manifest_indices=[s['manifest_index'] for s in samples],frozen_contexts=contexts)
