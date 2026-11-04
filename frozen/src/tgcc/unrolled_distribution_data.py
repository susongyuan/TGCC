"""Real contiguous demonstration blocks; no padding or cross-episode memory."""
from copy import deepcopy
import numpy as np
import torch
from .full_distribution_data import FullCorpusData, FullBatchPreparer
from .transition_features import current_outcome_from_context, outcome_mask, transition_target, normalize_transition
from .retrieval_adapter import RetrievalContext


def contiguous_windows(rows, blocks):
    if type(blocks) is not int or not 2 <= blocks <= 16:
        raise ValueError('Memory training needs 2..16 real contiguous action blocks')
    lookup={}
    for index,row in enumerate(rows):
        if row['split']!='train':continue
        key=(str(row['episode_id']),int(row['step_id']))
        if key in lookup:raise ValueError('Duplicate real demonstration start')
        lookup[key]=index
    windows=[]
    for (episode,start),index in sorted(lookup.items()):
        chain=[lookup.get((episode,start+32*k)) for k in range(blocks)]
        if any(i is None for i in chain):continue
        first=rows[index]
        for k,i in enumerate(chain):
            row=rows[i]
            if (row['scene_id']!=first['scene_id'] or row['dataset_root']!=first['dataset_root']
                    or row['source_sha256']!=first['source_sha256']
                    or row['sequence_steps']!=[start+32*k,start+32*(k+1)]
                    or abs(row['timestamp']-first['timestamp']-k*32/30)>2e-4):
                raise ValueError('Unroll crossed a source, scene or real timestamp boundary')
        windows.append(chain)
    if not windows:raise ValueError('No complete contiguous memory training windows')
    return windows


class UnrolledCorpusData:
    def __init__(self,manifest,base,plan,expected_manifest_sha256,blocks=8,burnin_blocks=4,exposure_cache_size=64):
        self.source=FullCorpusData(manifest,base,plan,expected_manifest_sha256)
        if type(exposure_cache_size) is not int or not 0<=exposure_cache_size<=256:
            raise ValueError('Bounded immutable CPU exposure cache required')
        self.source.exposure_cache_size=exposure_cache_size
        self.demonstrations=self.source.demonstrations
        self.blocks=blocks;self.windows=contiguous_windows(self.source.rows,blocks)
        if type(burnin_blocks) is not int or not 0<=burnin_blocks<=16:
            raise ValueError('Bounded real memory burn-in required')
        self.burnin_blocks=burnin_blocks
        self.lookup={(str(row['episode_id']),row['step_id']):i for i,row in enumerate(self.source.rows) if row['split']=='train'}
        self.rows=[dict(self.source.rows[chain[0]],memory_block_indices=chain,
                        memory_unroll_blocks=blocks) for chain in self.windows]
        self.train_indices=list(range(len(self.rows)))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        chain=self.windows[index];items=[self.source[i] for i in chain]
        sequence=[s['policy']['sequence'][0] for s in items]
        for first,second in zip(sequence,sequence[1:]):
            if (first['episode_id']!=second['episode_id'] or
                    abs(second['timestamp']-first['timestamp']-32/30)>2e-4):
                raise ValueError('Real memory exposures are not contiguous')
        policy=dict(items[0]['policy'],sequence=sequence)
        row=self.source.rows[chain[0]];prior=[]
        for k in range(1,self.burnin_blocks+1):
            key=(str(row['episode_id']),row['step_id']-32*k)
            if key not in self.lookup:break
            earlier_index=self.lookup[key];earlier=self.source.rows[earlier_index]
            if (any(earlier[name]!=row[name] for name in ('scene_id','dataset_root','source_sha256'))
                    or earlier['sequence_steps']!=[row['step_id']-32*k,row['step_id']-32*(k-1)]
                    or abs(row['timestamp']-earlier['timestamp']-k*32/30)>2e-4):
                raise ValueError('Memory burn-in crossed a source or real timestamp boundary')
            prior.append(earlier_index)
        burnin=[self.source[i]['policy']['sequence'][0] for i in reversed(prior)]
        if burnin and (burnin[-1]['episode_id']!=sequence[0]['episode_id'] or
                abs(sequence[0]['timestamp']-burnin[-1]['timestamp']-32/30)>2e-4):
            raise ValueError('Memory burn-in did not end immediately before training')
        burnin_policy=dict(policy,sequence=burnin) if burnin else None
        return dict(policy=policy,outcome_observation=items[-1]['policy']['sequence'][1]['observation'],
            policy_burnin=burnin_policy,
            raw_contexts=np.stack([s['raw_contexts'][0] for s in items]),
            raw_afters=np.stack([s['raw_after'] for s in items]),
            raw_after_masks=np.stack([s['raw_after_mask'] for s in items]),
            actions=np.stack([s['actions'] for s in items]),
            next_indices=np.asarray([s['next_index'] for s in items],np.int64),
            next_masks=np.asarray([s['next_mask'] for s in items],bool),
            episode_id=items[0]['episode_id'],manifest_index=index,
            original_manifest_indices=chain,memory_unroll_blocks=self.blocks)


TASK_PROMPT = "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked."
STATE_SLICE = slice(256, 279)  # standardized context: frozen visual PCA (256) then the official normalized state (23)


def perturbation_plan(seed, manifest_index, blocks, spec, state_width=23):
    """Deterministic per-window plan: which blocks are perturbed and the state noise.

    Depends only on (seed, window, block), never on rank, order or time, so an exact resume
    reproduces it. Unperturbed blocks carry exact zeros. The critic path never sees it.
    """
    if not spec: return np.zeros(blocks, bool), np.zeros((blocks, state_width), np.float32)
    fraction = float(spec['block_fraction']); state_std = float(spec['state_noise_std'])
    if not (0 <= fraction <= 1 and 0 < state_std <= 1):
        raise ValueError('Perturbation spec must be a bounded fraction and a normalized std in (0,1]')
    rng = np.random.default_rng([int(seed), int(manifest_index), 20261031])
    perturbed = rng.random(blocks) < fraction
    state_noise = rng.normal(0, state_std, (blocks, state_width)).astype(np.float32)
    state_noise[~perturbed] = 0
    return perturbed, state_noise


def correction_target(clean, noise):
    """Perturbed normalized state (clipped to the official [-1,1] range) in the state's own dtype, and the
    actual correction delta = clean - clip(clean + noise) as float32. Clipping makes the delta smaller than the noise."""
    clean = np.asarray(clean); perturbed = np.clip(clean + np.asarray(noise, clean.dtype), -1, 1).astype(clean.dtype)
    return perturbed, (clean - perturbed).astype(np.float32)


def official_state_tokenizer():
    """The official pi05 prompt tokenizer with the pi05 max token length (200), without importing the
    training config module (which drags in simulator dependencies)."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models.pi0_config import Pi0Config
    return PaligemmaTokenizer(Pi0Config(pi05=True, action_horizon=32).max_token_len)


def visual_content_key(obs):
    """SHA-1 of the three official camera images (uint8 bytes, fixed camera order)."""
    import hashlib
    from .retrieval_data import CAMERA_KEYS
    h = hashlib.sha1()
    for key in CAMERA_KEYS:
        value = obs['image'][key]
        value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


class UnrolledBatchPreparer(FullBatchPreparer):
    perturbation = None
    perturbation_seed = None
    _tokenizer = None
    visual_cache = None  # content-keyed cache of the frozen retrieval encoder outputs; None disables it
    visual_cache_stats = None

    def encode_visual(self, observations):
        """Frozen encoder outputs keyed by the exact camera pixels: identical images reuse the stored features
        bit-for-bit; state perturbation never touches images, so the key is unaffected by it."""
        if self.visual_cache is None:
            return self.encoder(observations)
        keys = [visual_content_key(obs) for obs in observations]
        missing = sorted({k for k in keys if k not in self.visual_cache})
        if missing:
            first = {k: next(i for i, kk in enumerate(keys) if kk == k) for k in missing}
            fresh = self.encoder([observations[first[k]] for k in missing])
            for k, row in zip(missing, fresh): self.visual_cache[k] = np.array(row, np.float32, copy=True)
        stats = self.visual_cache_stats
        if stats is not None:
            stats['hits'] += len(keys) - len(missing); stats['misses'] += len(missing); stats['calls'] += 1
            if stats['calls'] % 100 == 0: print(__import__('json').dumps(dict(stage='visual_cache', entries=len(self.visual_cache), **stats)), flush=True)
        return np.stack([self.visual_cache[k] for k in keys], 0)

    def retokenize(self, state32):
        if self._tokenizer is None: self._tokenizer = official_state_tokenizer()
        tokens, mask = self._tokenizer.tokenize(TASK_PROMPT, np.asarray(state32))  # keep the stored dtype; never cast before digitizing
        return np.asarray(tokens), np.asarray(mask)

    def perturb_item(self, item, noise):
        """Return (item, delta, effective, perturbed23, skipped). The cached item is never modified.
        If the official tokenization of the stored state cannot be reproduced bit-for-bit, the block is left clean
        (no perturbation, zero correction target, effective=False) and reported as skipped instead of aborting training."""
        obs = item['observation']; state = obs['state']; clean = state[0].detach().cpu().numpy()  # keep the stored dtype (float64): the official tokenizer digitized this exact array
        clean_tokens = obs['tokenized_prompt'][0].detach().cpu().numpy(); clean_mask = obs['tokenized_prompt_mask'][0].detach().cpu().numpy()
        # The official pipeline tokenizes the 23-dim normalized state; zero padding to 32 happens afterwards.
        check_tokens, check_mask = self.retokenize(clean[:23])
        if not (np.array_equal(check_tokens, clean_tokens) and np.array_equal(check_mask, clean_mask)):
            __import__('logging').getLogger('tgcc.perturbation').warning('state tokenization not reproducible bit-for-bit; block left clean (skipped)')
            return item, np.zeros(23, np.float32), False, clean[:23].copy(), True
        perturbed32 = clean.copy(); perturbed23, delta = correction_target(clean[:23], noise); perturbed32[:23] = perturbed23
        if clean.dtype != np.float64: raise ValueError('Expected the official float64 normalized state; refusing to perturb a differently typed state')
        tokens, mask = self.retokenize(perturbed23)
        effective = not np.array_equal(tokens, clean_tokens)
        new_obs = dict(obs, state=torch.as_tensor(perturbed32[None], dtype=state.dtype, device=state.device),
                       tokenized_prompt=torch.as_tensor(tokens[None], dtype=obs['tokenized_prompt'].dtype, device=obs['tokenized_prompt'].device),
                       tokenized_prompt_mask=torch.as_tensor(mask[None], dtype=obs['tokenized_prompt_mask'].dtype, device=obs['tokenized_prompt_mask'].device))
        return dict(item, observation=new_obs), delta, effective, perturbed23, False

    def __call__(self,samples):
        batch=len(samples);blocks=samples[0]['memory_unroll_blocks']
        if any(s['memory_unroll_blocks']!=blocks for s in samples):
            raise ValueError('Mixed configured memory lengths')
        burnin_lengths=[len(s['policy_burnin']['sequence']) if s.get('policy_burnin') else 0 for s in samples]
        if len(set(burnin_lengths))!=1:
            raise ValueError('Use micro1 or group equal real burn-in lengths; never pad memory exposures')
        observations=[obs for s in samples for obs in
            [*[x['observation'] for x in s['policy']['sequence']],s['outcome_observation']]]
        visual=self.encode_visual(observations).reshape(batch,blocks+1,3456)
        contexts=[];targets=[];masks=[];a=self.codec.artifact
        for i,sample in enumerate(samples):
            projected=self.codec.visual(visual[i])
            raw_context=np.concatenate([projected[:-1],sample['raw_contexts'][:,3456:]],-1)
            raw_after=np.concatenate([projected[1:],sample['raw_afters'][:,3456:]],-1)
            before=current_outcome_from_context(raw_context,projected.shape[-1])
            after_mask=np.concatenate([np.ones((blocks,projected.shape[-1]),bool),sample['raw_after_masks'][:,3456:]],-1)
            change,known=transition_target(before,raw_after,outcome_mask(before,projected.shape[-1]),after_mask)
            normalized,known=normalize_transition(change,known,a)
            targets.append(normalized)
            masks.append(known)
            contexts.append((raw_context-a['context_mean'])/a['context_std'])
        contexts=np.asarray(contexts,np.float32)
        def tensor(x,dtype=None):return torch.as_tensor(np.asarray(x),device=self.device,dtype=dtype)
        density=dict(context=tensor(contexts.reshape(batch*blocks,-1),torch.float32),
            actions=tensor(np.stack([s['actions'] for s in samples]).reshape(batch*blocks,32,23),torch.float32),
            action_mask=torch.ones(batch*blocks,32,device=self.device,dtype=torch.bool),
            after_features=tensor(np.asarray(targets).reshape(batch*blocks,-1),torch.float32),
            after_feature_mask=tensor(np.asarray(masks).reshape(batch*blocks,-1),torch.bool),
            next_index=tensor(np.stack([s['next_indices'] for s in samples]).reshape(-1),torch.long),
            planning_valid=tensor(np.stack([s['next_masks'] for s in samples]).reshape(-1),torch.bool),
            action_valid=torch.ones(batch*blocks,device=self.device,dtype=torch.bool),
            execution_valid=torch.ones(batch*blocks,device=self.device,dtype=torch.bool))
        samples=list(samples);queries=contexts.copy();perturbed=np.zeros((batch,blocks),bool);effective=np.zeros((batch,blocks),bool);skipped=np.zeros((batch,blocks),bool)
        correction=np.zeros((blocks,batch,23),np.float32)
        if self.perturbation:
            mean=a['context_mean'][STATE_SLICE];std=a['context_std'][STATE_SLICE]
            for i,sample in enumerate(samples):
                flags,noise=perturbation_plan(self.perturbation_seed,sample['manifest_index'],blocks,self.perturbation)
                sequence=list(sample['policy']['sequence'])
                for k in range(blocks):
                    if not flags[k]:continue
                    item,delta,changed,perturbed23,skip=self.perturb_item(sequence[k],noise[k])
                    sequence[k]=item;perturbed[i,k]=not skip;effective[i,k]=changed;skipped[i,k]=skip;correction[k,i]=delta
                    # The retrieval query is the same perturbed state passed through the frozen codec's state slice.
                    queries[i,k,STATE_SLICE]=(perturbed23.astype(np.float32)-mean)/std
                samples[i]=dict(sample,policy=dict(sample['policy'],sequence=sequence))
        references=[];ids=[]
        for k in range(blocks):
            records=[self.library.query_fast(queries[i,k],exclude_episode_id=s['episode_id'])[0][0]
                     for i,s in enumerate(samples)]
            if any(r['split']!='train' or str(r['episode_id'])==s['episode_id'] for r,s in zip(records,samples)):
                raise ValueError('Memory-window reference leaked its query episode')
            references.append(RetrievalContext(
                tensor([self.normalizer.transform(r['actual_native']) for r in records],torch.float32),
                tensor([r['action_mask'] for r in records],torch.bool),
                tensor([r['context'] for r in records],torch.float32),
                tensor([r['after_features'] for r in records],torch.float32),
                torch.ones(batch,device=self.device,dtype=torch.bool),
                tensor([r['after_feature_mask'] for r in records],torch.bool)))
            ids.append([str(r['episode_id']) for r in records])
        return dict(density=density,policy_samples=[s['policy'] for s in samples],references=references,
            policy_burnin_samples=[s['policy_burnin'] for s in samples] if burnin_lengths[0] else [],
            query_episode_ids=[s['episode_id'] for s in samples],reference_episode_ids=ids,
            manifest_indices=[s['manifest_index'] for s in samples],frozen_contexts=contexts,
            perturbed_blocks=torch.as_tensor(perturbed,device=self.device),perturbation_effective=torch.as_tensor(effective,device=self.device),perturbation_skipped=torch.as_tensor(skipped,device=self.device),
            correction_target=torch.as_tensor(correction,device=self.device),perturbed_contexts=tensor(queries.reshape(batch*blocks,-1),torch.float32))
