#!/usr/bin/env python3
"""Independent, explicitly incomplete CPU interface library from finished caches."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from tgcc.retrieval_data import digest,RetrievalDataset


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    sampling=json.loads((args.source/'sampling_plan.json').read_text())
    records=[];parts={};counts={'train':0,'validation':0};cache_identity={}
    for path in sorted(args.source.glob('raw_episode_*.npz')):
        rowpath=path.with_suffix('.rows.json')
        if not rowpath.exists():continue
        rows=json.loads(rowpath.read_text());split=rows[0]['split']
        if counts[split]>={'train':8,'validation':2}[split]:continue
        if len(rows)!=32:raise ValueError('Only finished whole-episode anchor caches are admitted')
        with np.load(path,allow_pickle=False) as f:
            for key in f.files:parts.setdefault(key,[]).append(f[key])
        records.extend(rows);counts[split]+=1
        cache_identity[path.name]=digest(path);cache_identity[rowpath.name]=digest(rowpath)
    if counts!={'train':8,'validation':2}:raise ValueError('Need 8 train and 2 validation cache episodes')
    arrays={k:np.concatenate(v) for k,v in parts.items()}
    train=np.asarray([r['split']=='train' for r in records]);val=~train
    visual=np.concatenate((arrays['raw_visual'][train],arrays['raw_after_visual'][train]))
    torch.manual_seed(20261027);v=torch.from_numpy(visual);mean=v.mean(0)
    _,_,components=torch.pca_lowrank(v-mean,q=256,center=False,niter=3)
    components=components.numpy();visual_mean=mean.numpy();width=visual.shape[1]
    project=lambda value:(value-visual_mean)@components
    context=np.concatenate((project(arrays['raw_context'][:,:width]),arrays['raw_context'][:,width:]),axis=1)
    after=np.concatenate((project(arrays['raw_after_features'][:,:width]),arrays['raw_after_features'][:,width:]),axis=1)
    mask=np.concatenate((np.ones((len(after),256),bool),arrays['raw_after_feature_mask'][:,width:]),axis=1)
    cm=context[train].mean(0);cs=np.maximum(context[train].std(0),.01)
    count=mask[train].sum(0);am=(after[train]*mask[train]).sum(0)/np.maximum(count,1)
    variance=((after[train]-am)**2*mask[train]).sum(0)/np.maximum(count,1);astd=np.maximum(np.sqrt(variance),.01)
    np.savez(args.output/'codec.npz',visual_mean=visual_mean,visual_components=components,
             context_mean=cm.astype(np.float32),context_std=cs.astype(np.float32),
             after_mean=am.astype(np.float32),after_std=astd.astype(np.float32))
    final={k:arrays[k] for k in ('actions','action_mask','actual_native','next_index','next_mask','reference_state','reference_after_state')}
    final.update(context=((context-cm)/cs).astype(np.float32),after_features=((after-am)/astd).astype(np.float32),
                 after_feature_mask=mask,reference_visual_context=project(arrays['raw_visual']).astype(np.float32))
    final['after_features'][~mask]=0
    np.savez_compressed(args.output/'samples.npz',**final)
    (args.output/'rows.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
    for filename in ('vision_encoder.safetensors','vision_config.json'):shutil.copyfile(args.source/filename,args.output/filename)
    artifacts={n:digest(args.output/n) for n in ('samples.npz','rows.jsonl','codec.npz')}
    import hashlib
    identity=dict(sampling['identity']);identity.update(temporary_incomplete=True,formal_library_ready=False,
        context_width=context.shape[1],after_width=after.shape[1],visual_width=256,
        train_samples=int(train.sum()),validation_samples=int(val.sum()),
        selected_train_episodes=8,selected_validation_episodes=2,
        index_episodes=sorted({r['episode_id'] for r in records if r['split']=='train'}),
        validation_episodes=sorted({r['episode_id'] for r in records if r['split']=='validation'}),
        pca_fit_episodes='train_only',pca_after_states_used_only_in_train_basis_fit=True,
        artifact_sha256=artifacts,codec_sha256=artifacts['codec.npz'],feature_codec_sha256=artifacts['codec.npz'],
        feature_artifact_sha256=artifacts,retrieval_index_sha256=hashlib.sha256((artifacts['samples.npz']+artifacts['rows.jsonl']).encode()).hexdigest(),
        vision_encoder_sha256=digest(args.output/'vision_encoder.safetensors'),vision_config_sha256=digest(args.output/'vision_config.json'),
        smoke_builder_sha256=digest(__file__),source_raw_cache_sha256=cache_identity,
        valid_training_targets=dict(action=int(train.sum()),execution=int(train.sum()),planning=int(final['next_mask'][train].sum())),
        retrieval_distance_groups=[dict(name='visual',slices=[[0,256]],weight=.20),
            dict(name='current_robot_and_calibration',slices=[[256,279],[326,467]],weight=.25),
            dict(name='current_three_graphs',slices=[[467,1565]],weight=.30),
            dict(name='three_graph_temporal_history',slices=[[1565,3050]],weight=.15),
            dict(name='executed_action_and_robot_history',slices=[[279,326],[3050,3097]],weight=.10)])
    (args.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    loaded=RetrievalDataset(args.output)
    for i in range(len(loaded)):
        reference,_=loaded.query(loaded.arrays['context'][i],exclude_episode_id=loaded.rows[i]['episode_id'])[0]
        if reference['split']!='train' or reference['episode_id']==loaded.rows[i]['episode_id']:raise ValueError('Smoke reference leakage')
    report=dict(status='passed',temporary_incomplete=True,formal_library_ready=False,samples=len(loaded),
                train_samples=len(loaded.train_indices),validation_samples=len(loaded.validation_indices),
                identity_sha256=digest(args.output/'identity.json'),whole_episode_exclusion_queries_checked=len(loaded))
    (args.output/'build_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
