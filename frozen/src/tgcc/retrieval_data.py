"""Success-demonstration support data without privileged simulator inputs.

The frozen codec reads three current RGB views, checkpoint-normalized robot
feedback, and actual past actions. Source labels and after states are targets or
retrieved trajectory content, never current-context inputs. Normal support does
not prove arbitrary failure recognition or task completion.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CAMERA_KEYS=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
VISION_PREFIX='base.paligemma_with_expert.paligemma.model.vision_tower.'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class FrozenVisualEncoder:
    """Independent step5000 SigLIP; remains fixed during joint policy updates."""
    def __init__(self,model,device='cpu'):
        self.model=model.to(device).eval().requires_grad_(False)
        self.device=torch.device(device)

    @classmethod
    def from_parent(cls,path,expected_sha,device='cpu'):
        from safetensors import safe_open
        from transformers import CONFIG_MAPPING
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        path=Path(path)
        if digest(path)!=expected_sha:raise ValueError('Frozen visual parent identity mismatch')
        config=CONFIG_MAPPING['paligemma']().vision_config
        config.intermediate_size=4304
        config.projection_dim=2048
        config.projector_hidden_act='gelu_fast'
        model=SiglipVisionModel(config)
        with safe_open(path,framework='pt',device='cpu') as f:
            weights={k[len(VISION_PREFIX):]:f.get_tensor(k) for k in f.keys() if k.startswith(VISION_PREFIX)}
        if not weights:raise ValueError('Parent has no owned vision weights')
        model.load_state_dict(weights,strict=True,assign=True)
        encoder=cls(model,device)
        encoder.config=config.to_dict()
        encoder.parent_sha256=expected_sha
        return encoder

    def save(self,directory):
        from safetensors.torch import save_file
        directory=Path(directory)
        path=directory/'vision_encoder.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in self.model.state_dict().items()},str(path))
        (directory/'vision_config.json').write_text(json.dumps(self.config,indent=2)+'\n')
        return dict(vision_encoder_sha256=digest(path),vision_config_sha256=digest(directory/'vision_config.json'))

    @classmethod
    def from_library(cls,directory,device='cpu'):
        from safetensors.torch import load_file
        from transformers import SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        directory=Path(directory)
        identity=json.loads((directory/'identity.json').read_text())
        for name in ('vision_encoder','vision_config'):
            path=directory/(name+('.safetensors' if name=='vision_encoder' else '.json'))
            if digest(path)!=identity[name+'_sha256']:raise ValueError('Visual codec artifact changed')
        config=SiglipVisionConfig(**json.loads((directory/'vision_config.json').read_text()))
        model=SiglipVisionModel(config)
        model.load_state_dict(load_file(str(directory/'vision_encoder.safetensors')),strict=True,assign=True)
        result=cls(model,device);result.config=config.to_dict();return result

    @torch.inference_mode()
    def __call__(self,observations):
        observations=observations if isinstance(observations,list) else [observations]
        pixels=[]
        for obs in observations:
            if set(obs['image'])!=set(CAMERA_KEYS):raise ValueError('Exactly three audited current cameras required')
            for key in CAMERA_KEYS:
                value=obs['image'][key]
                if not isinstance(value,torch.Tensor):value=torch.as_tensor(value)
                if value.shape!=(1,224,224,3) or value.dtype!=torch.uint8:
                    raise ValueError('Official pre-model uint8 RGB shape required')
                if not bool(torch.as_tensor(obs['image_mask'][key]).all()):raise ValueError('Missing visual exposure')
                # Exactly the official Observation.from_dict uint8 path.
                pixels.append(value.to(self.device,torch.float32).permute(0,3,1,2)/255*2-1)
        x=torch.cat(pixels,0)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            result=self.model(pixel_values=x).last_hidden_state.float().mean(1)
        result=torch.nn.functional.normalize(result,dim=-1)
        return result.reshape(len(observations),-1).cpu().numpy().astype(np.float32)


def past_action_summary(actions,frame,normalizer):
    """No action at frame or later can enter the current context."""
    actions=np.asarray(actions,np.float32)
    if actions.ndim!=2 or actions.shape[1]!=23 or not 0<=frame<=len(actions):
        raise ValueError('Invalid real action history')
    prior=actions[max(0,frame-32):frame]
    if not len(prior):return np.zeros(46,np.float32),False
    norm=np.asarray(normalizer.transform(prior),np.float32)
    if not np.isfinite(norm).all():raise ValueError('Nonfinite past action history')
    return np.concatenate([norm.mean(0),norm[-1]]),True


class FrozenFeatureCodec:
    def __init__(self,artifact):
        self.artifact={k:np.asarray(v) for k,v in artifact.items()}

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'codec.npz'
        identity=json.loads((Path(directory)/'identity.json').read_text())
        if identity.get('context_width') == 3097:
            from .transition_features import FEATURE_SCHEMA
            if identity.get('feature_schema') != FEATURE_SCHEMA:
                raise ValueError('Old absolute-outcome feature library: rebuild corrected library')
        if digest(path)!=identity['codec_sha256']:raise ValueError('Frozen PCA identity mismatch')
        with np.load(path,allow_pickle=False) as f:return cls({k:f[k] for k in f.files})

    def visual(self,raw):
        a=self.artifact;raw=np.asarray(raw,np.float32)
        return (raw-a['visual_mean'])@a['visual_components']

    def context(self,raw_visual,state,past_summary,past_valid):
        state=np.asarray(state,np.float32)
        if state.shape[-1]!=32 or np.any(state[...,23:]):raise ValueError('Official state32 padding required')
        x=np.concatenate([self.visual(raw_visual),state[...,:23],np.asarray(past_summary,np.float32),
                          np.asarray(past_valid,np.float32)[...,None]],axis=-1)
        a=self.artifact
        return ((x-a['context_mean'])/a['context_std']).astype(np.float32)

    def after(self,raw_visual,state):
        state=np.asarray(state,np.float32)
        if state.shape[-1]!=32 or np.any(state[...,23:]):raise ValueError('Official state32 padding required')
        x=np.concatenate([self.visual(raw_visual),state[...,:23]],axis=-1)
        a=self.artifact
        return ((x-a['after_mean'])/a['after_std']).astype(np.float32)


class RetrievalDataset:
    """Immutable support tensors and opaque provenance, train/val source-isolated."""
    def __init__(self,directory):
        self.directory=Path(directory)
        self.identity=json.loads((self.directory/'identity.json').read_text())
        if self.identity.get('context_width')==3097:
            from .transition_features import FEATURE_SCHEMA,EXECUTION_TARGET
            if (self.identity.get('feature_schema')!=FEATURE_SCHEMA or
                    self.identity.get('execution_target')!=EXECUTION_TARGET):
                raise ValueError('Old absolute-outcome support cannot enter corrected training/inference')
        for name in ('samples.npz','rows.jsonl','codec.npz'):
            if digest(self.directory/name)!=self.identity['artifact_sha256'][name]:
                raise ValueError('Retrieval artifact changed: '+name)
        self.rows=[json.loads(s) for s in (self.directory/'rows.jsonl').read_text().splitlines() if s]
        with np.load(self.directory/'samples.npz',allow_pickle=False) as f:
            self.arrays={k:f[k] for k in f.files}
        if self.identity.get('context_width')==3097:
            with np.load(self.directory/'codec.npz',allow_pickle=False) as f:
                supported=f['after_supported']
            if np.any(self.arrays['after_feature_mask'] & ~supported):
                raise ValueError('Library/calibration targets exceed train-supported execution space')
        if any(len(v)!=len(self.rows) for v in self.arrays.values()):raise ValueError('Support tensor/row count mismatch')
        ownership={};seen=set()
        for row in self.rows:
            ep=row['episode_id']
            if ownership.setdefault(ep,row['split'])!=row['split']:raise ValueError('Episode holdout leakage')
            key=(ep,row['sequence_steps'][0])
            if key in seen:raise ValueError('Duplicate support anchor')
            seen.add(key)
            if row['sequence_steps'][1]!=row['sequence_steps'][0]+32:
                raise ValueError('Support requires a real exact after exposure')
            if abs(row['after_timestamp']-row['before_timestamp']-32/30)>2e-4:
                raise ValueError('Unaligned support transition')
        self.train_indices=np.asarray([i for i,r in enumerate(self.rows) if r['split']=='train'],np.int64)
        self.validation_indices=np.asarray([i for i,r in enumerate(self.rows) if r['split']=='validation'],np.int64)
        if not len(self.train_indices) or not len(self.validation_indices):raise ValueError('Missing train/heldout source')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=deepcopy(self.rows[index])
        row.update({k:v[index].copy() for k,v in self.arrays.items()})
        # The exclusion is the whole episode, not merely a matching anchor.
        row['retrieval_excluded_episode_ids']=[row['episode_id']]
        return row

    def query(self,context,k=1,exclude_episode_id=None,eligible_next=None):
        query=np.asarray(context,np.float32)
        if query.shape!=(self.identity['context_width'],) or not np.isfinite(query).all():
            raise ValueError('Query must use the frozen current/past codec')
        allowed=self.train_indices
        if exclude_episode_id is not None:
            allowed=np.asarray([i for i in allowed if self.rows[i]['episode_id']!=str(exclude_episode_id)],np.int64)
        if eligible_next is not None:
            eligible=np.asarray(eligible_next,bool)
            valid=self.arrays['next_mask'][allowed]
            targets=self.arrays['next_index'][allowed]
            allowed=allowed[valid & eligible[targets]]
        if len(allowed)<k:raise ValueError('Insufficient episode-excluded reference support')
        x=self.arrays['context'][allowed]
        distances, _ = self.distance_components(query,x)
        chosen=np.argsort(distances,kind='stable')[:k]
        return [(self[int(allowed[j])],float(distances[j])) for j in chosen]

    def distance_components(self,query,candidates):
        query=np.asarray(query,np.float32);candidates=np.asarray(candidates,np.float32)
        groups=self.identity.get('retrieval_distance_groups')
        if not groups:raise ValueError('Explicit state/vision/graph/history distance groups are required')
        pieces={};total=np.zeros(len(candidates),np.float32);weight=0.
        for group in groups:
            columns=np.concatenate([np.arange(lo,hi) for lo,hi in group['slices']])
            if not len(columns) or np.any(columns>=self.identity['context_width']):
                raise ValueError('Invalid retrieval feature group')
            pieces[group['name']]=np.mean((candidates[:,columns]-query[columns])**2,axis=-1)
            total+=float(group['weight'])*pieces[group['name']];weight+=float(group['weight'])
        if abs(weight-1)>1e-6:raise ValueError('Retrieval group weights must sum to one')
        return total,pieces

    def retrieval_context(self,context,exclude_episode_id=None,eligible_next=None):
        sample,distance=self.query(context,exclude_episode_id=exclude_episode_id,eligible_next=eligible_next)[0]
        return dict(reference_actions=sample['actual_native'][None],
                    reference_state=sample['reference_state'][None],
                    reference_after_state=sample['reference_after_state'][None],
                    reference_visual_context=sample['reference_visual_context'][None],
                    reference_context=sample['context'][None],reference_after_features=sample['after_features'][None],
                    reference_after_feature_mask=sample['after_feature_mask'][None],
                    action_mask=sample['action_mask'][None],context_mask=np.ones(1,bool),
                    source_id=f"{sample['episode_id']}:{sample['sequence_steps'][0]}",distance=distance)

# Extra measured feedback omitted by the official state23 input. Articulation
# root world pose is deliberately absent: it is not an onboard observation.
EXTRA_FIELDS=(('base_qpos',3),('trunk_qvel',4),('arm_left_qvel',7),('arm_right_qvel',7),
              ('gripper_left_qvel',2),('gripper_right_qvel',2),('eef_left_pos',3),
              ('eef_left_quat',4),('eef_right_pos',3),('eef_right_quat',4))
GOAL_OBJECTS=('hotdog.n.02_1','hotdog.n.02_2','electric_refrigerator.n.01_1','microwave.n.02_1')


def feedback_features(named):
    values=[];valid=[]
    for key,width in EXTRA_FIELDS:
        v=named.get(key)
        known=v is not None and np.asarray(v).shape==(width,) and np.isfinite(v).all()
        values.extend(np.asarray(v,np.float32).tolist() if known else [0.]*width)
        valid.extend([known]*width)
    return np.asarray(values,np.float32),np.asarray(valid,bool)


def side_information(named,cam_rel_poses=None,depth_available=None):
    extra,valid=feedback_features(named)
    camera=np.asarray(cam_rel_poses,np.float32).reshape(-1) if cam_rel_poses is not None else np.zeros(21,np.float32)
    camera_valid=np.full(21,cam_rel_poses is not None,bool)
    if camera.shape!=(21,) or not np.isfinite(camera).all():raise ValueError('Expected calibrated 3-camera relative poses')
    depth=np.asarray(depth_available if depth_available is not None else [False]*3,np.float32)
    if depth.shape!=(3,):raise ValueError('Per-camera depth availability mask required')
    return np.concatenate([extra,valid.astype(np.float32),camera,camera_valid.astype(np.float32),depth])


def privileged_side_information(named,cam_rel_poses=None,depth_available=None):
    base=side_information(named,cam_rel_poses,depth_available)
    values=[];valid=[]
    for key in ('robot_pos','robot_ori_cos','robot_ori_sin'):
        x=named.get(key);known=x is not None and np.asarray(x).shape==(3,) and np.isfinite(x).all()
        values.extend(np.asarray(x,np.float32).tolist() if known else [0.]*3);valid.extend([known]*3)
    return np.r_[base,values,np.asarray(valid,np.float32)].astype(np.float32)


def raw_after_vector(raw_visual,state,*,named_proprio,cam_rel_poses=None,conditions,
                     depth_available=None,visual_project=None):
    v=visual_project(raw_visual) if visual_project else np.asarray(raw_visual,np.float32)
    return np.concatenate([v,np.asarray(state,np.float32)[:23],privileged_side_information(named_proprio,cam_rel_poses,depth_available),
                           production_graph_summary(conditions)]).astype(np.float32)


def _encode_upperbound_context(self,raw_visual,state,past_summary,past_valid,**kwargs):
    x=raw_context_vector(raw_visual,state,past_summary,past_valid,visual_project=self.visual,**kwargs)
    a=self.artifact;return ((x-a['context_mean'])/a['context_std']).astype(np.float32)


def _encode_upperbound_after(self,raw_visual,state,**kwargs):
    x=raw_after_vector(raw_visual,state,visual_project=self.visual,**kwargs)
    if 'after_supported' in self.artifact:
        # The controller binds this actual current observation to its preceding
        # observation before calculating a normalized execution change.
        return x.astype(np.float32)
    a=self.artifact;return ((x-a['after_mean'])/a['after_std']).astype(np.float32)

FrozenFeatureCodec.encode_context=_encode_upperbound_context
FrozenFeatureCodec.encode_after=_encode_upperbound_after

# Frozen v2 graph layout: role-wise metric facts, target-identity geometry
# moments, directed relation type/role/identity signatures. The same summary
# order is used at collection, training, and inference.
GRAPH_ROLE_FIELDS=np.asarray([0,1,*range(4,13),17,18,19,20],np.int64)
GRAPH_CURRENT_WIDTH=1098
GRAPH_HISTORY_WIDTH=1485


def _graph_frame(x,valid,roles,relations,name):
    active=x[valid].copy();rr=roles[valid].astype(np.int64)
    active[:,23:30]=0
    if name=='task_graph':
        active[:,17:23]=0
        # Evidence knownness differs between curated demos and online Oracle;
        # it is neither a physical target nor a retrieval progress signal.
        active[np.isin(rr,[9,10]),0]=0
    means=np.zeros((12,15),np.float32);ident=np.zeros((12,2),np.float32);counts=np.zeros(12,np.float32)
    for role in range(12):
        nodes=active[rr==role]
        if len(nodes):means[role]=nodes[:,GRAPH_ROLE_FIELDS].mean(0);ident[role]=nodes[:,30:32].mean(0);counts[role]=len(nodes)
    moments=np.zeros((2,9),np.float32)
    if len(active):moments=np.einsum('ni,nj->ij',active[:,30:32],active[:,4:13])/len(active)
    hist=np.zeros(32,np.float32);role_pairs=np.zeros(32,np.float32);id_rel=np.zeros((32,2),np.float32)
    ids=np.flatnonzero(valid)
    subset=relations[np.ix_(ids,ids)]
    source,target=np.nonzero(subset)
    for s,t in zip(source,target):
        relation=int(subset[s,t])
        if not 1<=relation<32:raise ValueError('Invalid directed relation vocabulary')
        hist[relation]+=1
        role_pairs[(int(rr[s])*12+int(rr[t]))%32]+=1
        # Ordered source/target identity signatures make swapping a relation
        # endpoint visible even if its relation-type count does not change.
        id_rel[relation]+=np.asarray([active[s,30],active[t,31]])
    divisor=max(1,len(source));hist/=divisor;role_pairs/=divisor;id_rel/=divisor
    meta=np.asarray([valid.mean(),np.mean(active[:,20]) if len(active) else 0,
                     np.mean(active[:,0]) if len(active) else 0,np.mean(active[:,1]) if len(active) else 0],np.float32)
    return np.r_[means.reshape(-1),moments.reshape(-1),ident.reshape(-1),counts/max(1,len(active)),
                  hist,role_pairs,id_rel.reshape(-1),meta].astype(np.float32),means[:,2:11].reshape(-1),moments.reshape(-1)


def production_graph_summary(conditions,*,historical=False,history_timestamps=None,timestamp=None):
    def array(v):return v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v)
    out=[]
    for name in ('global_graph','task_graph','local_graph'):
        graph=conditions[name] if isinstance(conditions,dict) else getattr(conditions,name)
        get=lambda k:graph[k] if isinstance(graph,dict) else getattr(graph,k)
        x=array(get('features'))[0];mask=array(get('valid'))[0].astype(bool);relations=array(get('relations'))[0]
        roles=graph.get('roles') if isinstance(graph,dict) else getattr(graph,'roles',None)
        roles=np.zeros(mask.shape,np.int64) if roles is None else array(roles)[0]
        if x.ndim!=3 or x.shape[-1]!=32 or mask.shape!=x.shape[:2]:raise ValueError('Production graph tensor schema')
        if not historical:
            out.append(_graph_frame(x[-1],mask[-1],roles[-1],relations[-1],name)[0]);continue
        if len(x)<2:
            out.append(np.zeros(495,np.float32));continue
        first,fg,fm=_graph_frame(x[0],mask[0],roles[0],relations[0],name)
        last,lg,lm=_graph_frame(x[-2],mask[-2],roles[-2],relations[-2],name)
        times=np.asarray(history_timestamps if history_timestamps is not None else [],np.float64)
        if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times)<=0) or timestamp is None or times[-1]>=timestamp):
            raise ValueError('Historical exposure timestamps must strictly precede current time')
        timing=[len(x)-1,float(timestamp-times[-1]) if len(times) else 0.,float(times[-1]-times[0]) if len(times)>1 else 0.]
        out.append(np.r_[last,lg-fg,lm-fm,timing].astype(np.float32))
    return np.concatenate(out)


def raw_context_vector(raw_visual,state,past_summary,past_valid,*,named_proprio,cam_rel_poses=None,
                       conditions,history_states=None,depth_available=None,visual_project=None,
                       timestamp=None,history_timestamps=None):
    v=visual_project(raw_visual) if visual_project else np.asarray(raw_visual,np.float32)
    past=np.asarray(history_states if history_states is not None else [],np.float32)
    if len(past):
        if past.ndim!=2 or past.shape[1]!=32 or np.any(past[:,23:]):raise ValueError('Past robot state32 schema')
        prior=np.r_[past[:,:23].mean(0),past[-1,:23],1.]
    else:prior=np.zeros(47,np.float32)
    from .transition_features import stable_scoring_context
    result=np.concatenate([v,np.asarray(state,np.float32)[:23],past_summary,[float(past_valid)],
            privileged_side_information(named_proprio,cam_rel_poses,depth_available),production_graph_summary(conditions),
            production_graph_summary(conditions,historical=True,history_timestamps=history_timestamps,timestamp=timestamp),prior]).astype(np.float32)
    return stable_scoring_context(result,len(v))


def after_feature_mask(named_proprio,conditions,*,visual_width=256,cam_rel_poses=None):
    """UNKNOWN target dimensions are never trained as measured zero values."""
    mask=np.ones(visual_width+23+141+GRAPH_CURRENT_WIDTH,bool)
    _,known=feedback_features(named_proprio)
    offset=visual_width+23
    mask[offset:offset+39]=known
    if cam_rel_poses is None:mask[offset+78:offset+99]=False
    for j,key in enumerate(('robot_pos','robot_ori_cos','robot_ori_sin')):
        if key not in named_proprio:mask[offset+123+3*j:offset+126+3*j]=False
    # Task-stage status slots retained in role summaries are intentionally UNKNOWN.
    # role fields17:20 are the final four entries, and were zeroed for task layer.
    graphstart=offset+141
    for role in range(12):mask[graphstart+366+role*15+11:graphstart+366+role*15+15]=False
    for layer,name in enumerate(('global_graph','task_graph','local_graph')):
        summary=production_graph_summary(conditions)[layer*366:(layer+1)*366]
        for role in range(12):
            base=graphstart+layer*366
            if summary[222+role]==0:
                mask[base+role*15:base+(role+1)*15]=False
                mask[base+198+role*2:base+200+role*2]=False
            elif summary[role*15+14]==0:
                # No verified metric extent for this role; its numerical zero
                # is UNKNOWN, not a measured zero-size object.
                mask[base+role*15+11:base+role*15+14]=False
        if not np.any(summary[234:266]):
            mask[graphstart+layer*366+234:graphstart+layer*366+362]=False
    from .transition_features import outcome_mask
    raw = np.r_[np.zeros(visual_width+23,np.float32),
        privileged_side_information(named_proprio,cam_rel_poses),production_graph_summary(conditions)]
    return mask & outcome_mask(raw,visual_width)
