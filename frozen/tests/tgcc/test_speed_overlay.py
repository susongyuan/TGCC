"""Speed overlay contracts: divisible micro-batches, exact burn-in grouping, RAM-only block cache, exact fast retrieval."""
import shutil
import numpy as np
import pytest
import torch


def test_micro_batch_must_divide_per_rank_windows():
    from tgcc.full_distribution_joint import full_accumulation
    cfg=dict(gradient_accumulation_by_world={'3':16,'4':12},max_gpus=4,effective_batch=48)
    assert full_accumulation(4,1,cfg)==12 and full_accumulation(4,2,cfg)==6 and full_accumulation(4,6,cfg)==2
    assert full_accumulation(4,12,cfg)==1 and full_accumulation(3,16,cfg)==1
    for world,micro in ((4,5),(4,8),(4,0),(4,24),(2,1),(5,1)):
        with pytest.raises(ValueError):full_accumulation(world,micro,cfg)


def test_burnin_groups_never_mix_lengths_and_preserve_order():
    from tgcc.full_distribution_joint import group_by_burnin
    def sample(n,tag):return dict(tag=tag,policy_burnin=(dict(sequence=[0]*n) if n else None))
    samples=[sample(4,'a'),sample(0,'b'),sample(4,'c'),sample(2,'d'),sample(4,'e')]
    groups=group_by_burnin(samples)
    assert [[x['tag'] for x in g] for g in groups]==[['b'],['d'],['a','c','e']]
    assert sum(len(g) for g in groups)==len(samples)


def test_group_share_scaling_equals_single_microbatch_mean():
    from tgcc.distribution_joint import effective_distribution_loss
    zero=dict(action=torch.tensor(0.),execution=torch.tensor(0.),planning=torch.tensor(0.))
    counts=dict(action=0,execution=0,planning=0);known=dict(action=1,execution=1,planning=1)
    weights=dict(action=1.,execution=1.,planning=1.,flow=1.,reference_improvement=.2)
    values=torch.tensor([1.,2.,3.,10.,20.]);hinge=torch.tensor([.1,.2,.3,.4,.5]);accumulation=2;n=len(values)
    whole=effective_distribution_loss(dict(zero,flow=values.mean(),reference_improvement=hinge.mean()),counts,known,4,accumulation,weights)
    total=torch.tensor(0.)
    for group in ([0,1,2],[3,4]):
        scale=accumulation*n/len(group)
        total=total+effective_distribution_loss(dict(zero,flow=values[group].mean(),reference_improvement=hinge[group].mean()),counts,known,4,scale,weights)
    assert torch.allclose(total,whole)


def test_block_cache_is_ram_only_identity_bound_and_exact(tmp_path):
    from tgcc.speed_cache import BlockCache,ram_backed
    if not ram_backed(tmp_path):
        with pytest.raises(ValueError):BlockCache(tmp_path/'cache',dict(code='x'),2**30,2)
    if not ram_backed('/dev/shm'):pytest.skip('no tmpfs available')
    root='/dev/shm/tgcc_speed_cache_pytest';shutil.rmtree(root,ignore_errors=True)
    try:
        identity=dict(manifest='m',plan='p',normalization='n',code='c')
        cache=BlockCache(root,identity,2**30,2)
        value=dict(x=np.arange(5,dtype=np.float32),t=torch.arange(3),s='ep',f=1.5,nested=dict(m=np.zeros((2,2),bool)),idx=7)
        assert cache.load('ep',7) is None
        cache.store('ep',7,value);back=cache.load('ep',7)
        assert np.array_equal(back['x'],value['x']) and back['x'].dtype==np.float32 and torch.equal(back['t'],value['t'])
        assert back['s']=='ep' and back['f']==1.5 and back['idx']==7 and np.array_equal(back['nested']['m'],value['nested']['m'])
        back['x'][0]=99.;assert cache.load('ep',7)['x'][0]==0.
        other=BlockCache(root,dict(identity,code='other'),2**30,2)
        assert other.load('ep',7) is None and other.root!=cache.root
        for episode in ('e1','e2','e3','e4'):
            for i in range(16):cache.store(episode,i,value)
        dirs={d.name for d in cache.root.iterdir() if d.is_dir()}
        assert 'e4' in dirs and len(dirs)<=2
        with pytest.raises(ValueError):BlockCache(root,identity,2**20,2)
    finally:shutil.rmtree(root,ignore_errors=True)


class _Library:
    def __init__(self):
        from tgcc.retrieval_data import RetrievalDataset
        rng=np.random.default_rng(0);n,width=60,12
        self.arrays=dict(context=rng.standard_normal((n,width)).astype(np.float32))
        self.arrays['context'][5]=self.arrays['context'][17]  # exact tie between two train rows
        self.rows=[dict(episode_id=str(i%5),split='train' if i%7 else 'validation',sequence_steps=[32*i,32*i+32]) for i in range(n)]
        self.train_indices=np.asarray([i for i,r in enumerate(self.rows) if r['split']=='train'],np.int64)
        self.identity=dict(context_width=width,retrieval_distance_groups=[dict(name='a',slices=[[0,4]],weight=.5),dict(name='b',slices=[[4,8],[10,12]],weight=.5)])
        self.query=RetrievalDataset.query.__get__(self);self.distance_components=RetrievalDataset.distance_components.__get__(self)
    def __getitem__(self,index):return dict(self.rows[index],index=index)


def test_fast_query_is_bit_identical_to_original_including_ties_and_exclusion():
    from tgcc.speed_cache import install_fast_query
    lib=_Library();install_fast_query(lib);ctx=lib.arrays['context']
    for q in list(lib.train_indices)+[5,17]:
        for exclude in (None,lib.rows[q]['episode_id']):
            for k in (1,3):
                a=lib.query(ctx[q],k=k,exclude_episode_id=exclude);b=lib.query_fast(ctx[q],k=k,exclude_episode_id=exclude)
                assert [r['index'] for r,_ in a]==[r['index'] for r,_ in b]
                assert [d for _,d in a]==[d for _,d in b]
                assert all(r['split']=='train' for r,_ in b)
                if exclude is not None:assert all(r['episode_id']!=exclude for r,_ in b)
    with pytest.raises(ValueError):lib.query_fast(np.zeros(11,np.float32))
    with pytest.raises(ValueError):lib.query_fast(np.full(12,np.nan,np.float32))
