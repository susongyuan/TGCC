"""Compare online full-corpus facts against immutable reference-library anchors."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'code/src'),str(ROOT/'code/packages/openpi-client/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
os.environ.setdefault('TGR_GEOMETRY_CACHE_PATH','/media/data5/ssy/TGR/cache/geometry_2f7d86e3609e0d9e5533b16ee67c4d03403bebe16470125fc1d99cd6babc5645.sqlite3')
from tgr.factory import ensure_legacy_imports
ensure_legacy_imports()
import numpy as np
import torch
from tgcc.full_distribution_data import FullCorpusData,FullBatchPreparer
from tgcc.checkpoint import digest,sources
from train import BASE


def main():
    p=argparse.ArgumentParser();p.add_argument('--feature-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_device(device)
    data=FullCorpusData(ROOT/'reports/hotdog_lazy_seq2_manifest.jsonl',BASE,ROOT/'reports/data_plan_vocabulary.json',
        digest(ROOT/'reports/hotdog_lazy_seq2_manifest.jsonl'))
    prepare=FullBatchPreparer(a.feature_library,BASE,device);records=[]
    for library_index in [int(prepare.library.train_indices[j]) for j in (0,2000,5887)]:
        anchor=prepare.library[library_index];sample=data[int(anchor['manifest_index'])];batch=prepare([sample]);density=batch['density']
        context=density['context'][0].cpu().numpy();after=density['after_features'][0].cpu().numpy()
        # Independent frozen SigLIP batching may produce tiny BF16 differences;
        # nonvisual causal facts/labels/masks must match exactly within FP32 rounding.
        for name,x,y in [('context_nonvisual',context[256:],anchor['context'][256:]),('after_nonvisual',after[256:],anchor['after_features'][256:])]:
            # The stored after array was standardized with FP64 masked
            # statistics; inference serializes those statistics to FP32.
            # Account for that documented roundoff, not semantic mismatch.
            np.testing.assert_allclose(x,y,rtol=3e-6,atol=3e-5 if name=='after_nonvisual' else 3e-6,err_msg=name)
        visual_error=float(np.max(np.abs(context[:256]-anchor['context'][:256])))
        after_visual_error=float(np.max(np.abs(after[:256]-anchor['after_features'][:256])))
        if max(visual_error,after_visual_error)>.08:raise AssertionError('Frozen visual projection drift exceeds BF16 batch tolerance')
        np.testing.assert_array_equal(density['after_feature_mask'][0].cpu().numpy(),anchor['after_feature_mask'])
        np.testing.assert_allclose(density['actions'][0].cpu().numpy(),anchor['actions'],atol=3e-6,rtol=2e-6)
        assert int(density['next_index'][0])==int(anchor['next_index']) and bool(density['planning_valid'][0])==bool(anchor['next_mask'])
        assert len(batch['references'])==2 and len(batch['reference_episode_ids'])==2
        for ids in batch['reference_episode_ids']:assert ids[0]!=str(anchor['episode_id'])
        assert not bool(prepare.encoder.model.training) and not any(p.requires_grad for p in prepare.encoder.model.parameters())
        records.append(dict(manifest_index=sample['manifest_index'],episode_id=sample['episode_id'],
            current_visual_max_error=visual_error,after_visual_max_error=after_visual_error,reference_episode_ids=batch['reference_episode_ids']))
        print(json.dumps(dict(stage='full_online_anchor_checked',**records[-1])),flush=True)
    anchors={int(row['manifest_index']) for row in prepare.library.rows if row['split']=='train'}
    nonanchor=next(i for i in data.train_indices if i not in anchors)
    batch=prepare([data[nonanchor]])
    assert batch['manifest_indices']==[nonanchor]
    rejected=False
    try:data[next(i for i,r in enumerate(data.rows) if r['split']=='validation')]
    except ValueError:rejected=True
    assert rejected
    result=dict(status='passed',full_data_gate_passed=True,source_sha256=sources(ROOT),
        feature_library_sha256=digest(a.feature_library/'identity.json'),train_sequences=51714,train_episodes=184,
        retrieval_train_anchors=5888,nonanchor_manifest_index=nonanchor,validation_imitation_rejected=True,
        both_exposures_retrieve=True,own_episode_excluded=True,unknown_after_mask_preserved=True,
        native_target_matches=True,independent_encoder_frozen=True,anchor_checks=records,
        source_success_caveat='curated human support; not every episode has an independent terminal success receipt')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    data.demonstrations.close();print(json.dumps(result),flush=True)


if __name__=='__main__':main()
