"""RAM-only sliding block cache and exact vectorized retrieval for faster data preparation.

Cached values are the unchanged outputs of the existing data code; a miss computes through
that code and stores the result. Entries are bound to the manifest, plan, normalization and
data-module source hashes. The root must be a RAM-backed filesystem (tmpfs): nothing is
written to disk. Eviction keeps only the most recently written episodes under a byte cap, so a
busy machine degrades to recomputation, never to wrong data.
"""
import hashlib,json,os,shutil,uuid
from pathlib import Path
import numpy as np
import torch

DATA_MODULES=('src/tgcc/full_distribution_data.py','src/tgcc/unrolled_distribution_data.py','src/tgcc/retrieval_data.py',
    'src/tgcc/transition_features.py','src/tgcc/robot.py','src/tgcc/speed_cache.py','src/tgcc/data.py')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def data_code_identity(root):
    root=Path(root);names=sorted(str(p.relative_to(root)) for p in (root/'src/tgr').glob('*.py'))+list(DATA_MODULES)
    return hashlib.sha256(json.dumps({n:_sha(root/n) for n in names},sort_keys=True).encode()).hexdigest()


def ram_backed(path):
    path=Path(path).resolve();best=None
    for line in Path('/proc/mounts').read_text().splitlines():
        parts=line.split()
        if len(parts)<3:continue
        mount=Path(parts[1])
        if mount==path or mount in path.parents:
            if best is None or len(str(mount))>len(str(best[0])):best=(mount,parts[2])
    return best is not None and best[1] in ('tmpfs','ramfs')


class BlockCache:
    def __init__(self,root,identity,cap_bytes,keep_episodes=3,prune=True):
        if not ram_backed(Path(root).parent if not Path(root).exists() else root):
            raise ValueError('Block cache root must be a RAM-backed filesystem; disk caching is refused')
        self.identity=dict(identity);name=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()[:24]
        self.root=Path(root)/name;self.root.mkdir(parents=True,exist_ok=True)
        marker=self.root/'identity.json'
        if not marker.exists():
            tmp=marker.with_suffix('.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(self.identity,indent=2,sort_keys=True)+'\n');os.replace(tmp,marker)
        if type(cap_bytes) is not int or not 2**30<=cap_bytes<=64*2**30:raise ValueError('Block cache cap must be 1..64 GiB')
        if type(keep_episodes) is not int or not 1<=keep_episodes<=8:raise ValueError('Keep 1..8 recent episodes')
        self.cap=cap_bytes;self.keep=keep_episodes;self.prune=bool(prune);self.hits=0;self.misses=0;self.stores=0

    def path(self,episode,index):return self.root/str(episode)/f'{int(index):06d}.pt'

    def load(self,episode,index):
        try:value=torch.load(self.path(episode,index),map_location='cpu',weights_only=False)
        except FileNotFoundError:self.misses+=1;return None
        except Exception:self.misses+=1;return None
        self.hits+=1;return value

    def store(self,episode,index,value):
        p=self.path(episode,index)
        if p.exists():return
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
        try:torch.save(value,tmp);os.replace(tmp,p)
        except OSError:
            try:tmp.unlink()
            except OSError:pass
            return
        self.stores+=1
        if self.prune and self.stores%16==0:self._prune(str(episode))

    def _prune(self,current):
        info=[]
        try:dirs=[d for d in self.root.iterdir() if d.is_dir()]
        except FileNotFoundError:return
        for d in dirs:
            try:
                files=[f for f in d.iterdir() if f.is_file()]
                size=sum(f.stat().st_size for f in files)
                mtime=max((f.stat().st_mtime for f in files),default=d.stat().st_mtime)
            except FileNotFoundError:continue
            info.append((mtime,size,d))
        info.sort(key=lambda x:x[0]);total=sum(s for _,s,_ in info)
        candidates=[x for x in info if x[2].name!=current]
        while candidates and (len(info)>self.keep or total>self.cap):
            mtime,size,d=candidates.pop(0);shutil.rmtree(d,ignore_errors=True);total-=size;info=[x for x in info if x[2]!=d]

    def stats(self):
        try:size=sum(f.stat().st_size for d in self.root.iterdir() if d.is_dir() for f in d.iterdir() if f.is_file())
        except FileNotFoundError:size=0
        return dict(root=str(self.root),hits=self.hits,misses=self.misses,stores=self.stores,bytes=size)


def install_fast_query(library):
    """Exact top-k with the original operation sequence. Per excluded episode the gathered
    candidate arrays (the same fresh contiguous copies the original builds for every query)
    are kept for the few most recent episodes, so only the subtract/square/mean remains."""
    groups=library.identity.get('retrieval_distance_groups')
    if not groups:raise ValueError('Explicit state/vision/graph/history distance groups are required')
    width=library.identity['context_width'];spec=[];weight=0.
    for group in groups:
        columns=np.concatenate([np.arange(lo,hi) for lo,hi in group['slices']])
        if not len(columns) or np.any(columns>=width):raise ValueError('Invalid retrieval feature group')
        spec.append((columns,float(group['weight'])));weight+=float(group['weight'])
    if abs(weight-1)>1e-6:raise ValueError('Retrieval group weights must sum to one')
    cache={};recent=[]
    def prepared(exclude_episode_id):
        key=None if exclude_episode_id is None else str(exclude_episode_id)
        if key in cache:return cache[key]
        allowed=library.train_indices
        if key is not None:
            allowed=np.asarray([i for i in allowed if library.rows[i]['episode_id']!=key],np.int64)
        x=library.arrays['context'][allowed]
        cache[key]=(allowed,[x[:,columns] for columns,_ in spec]);recent.append(key)
        while len(recent)>3:cache.pop(recent.pop(0),None)
        return cache[key]
    def query_fast(context,k=1,exclude_episode_id=None):
        query=np.asarray(context,np.float32)
        if query.shape!=(width,) or not np.isfinite(query).all():raise ValueError('Query must use the frozen current/past codec')
        allowed,parts=prepared(exclude_episode_id)
        if len(allowed)<k:raise ValueError('Insufficient episode-excluded reference support')
        total=np.zeros(len(allowed),np.float32)
        for (columns,w),candidates in zip(spec,parts):
            total+=w*np.mean((candidates-query[columns])**2,axis=-1)
        chosen=np.argsort(total,kind='stable')[:k]
        return [(library[int(allowed[j])],float(total[j])) for j in chosen]
    library.query_fast=query_fast;return library
