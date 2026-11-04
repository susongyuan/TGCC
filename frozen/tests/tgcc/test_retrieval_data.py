import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from tgcc.retrieval_data import (RetrievalDataset,production_graph_summary,past_action_summary,
                                after_feature_mask,_graph_frame)


def graph(features=None,roles=None,relations=None,times=1):
    x=np.zeros((1,times,3,32),np.float32)
    x[...,0]=1;x[...,20]=1;x[0,:,0,30:32]=[1,0];x[0,:,1,30:32]=[0,1];x[0,:,2,30:32]=[-1,0]
    if features is not None:x[0]=features
    r=np.broadcast_to(np.asarray(roles or [1,6,7],np.int64),(1,times,3)).copy()
    rel=np.zeros((1,times,3,3),np.int64)
    if relations is not None:rel[0]=relations
    return dict(features=x,roles=r,relations=rel,valid=np.ones((1,times,3),bool),oracle=True)


def conditions(g):return {k:g for k in ('global_graph','task_graph','local_graph')}


def test_target_position_swap_is_visible_even_when_untyped_mean_stays_same():
    a=graph(roles=[1,1,6]);a['features'][0,0,0,4]=1;a['features'][0,0,1,4]=3
    b={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in a.items()}
    b['features'][0,0,0,4]=3;b['features'][0,0,1,4]=1
    assert np.array_equal(a['features'].mean((1,2)),b['features'].mean((1,2)))
    assert not np.array_equal(production_graph_summary(conditions(a)),production_graph_summary(conditions(b)))


def test_left_right_and_directed_relations_change_the_same_query_codec():
    a=graph();a['features'][0,0,1,4]=1;a['features'][0,0,2,4]=3;a['relations'][0,0,1,0]=4
    b=graph();b['features'][:]=a['features'];b['relations'][0,0,2,0]=4
    assert not np.array_equal(production_graph_summary(conditions(a)),production_graph_summary(conditions(b)))
    c=graph();c['features'][:]=a['features'];c['relations'][:]=a['relations'];c['features'][0,0,[1,2],4]=[3,1]
    assert not np.array_equal(production_graph_summary(conditions(a)),production_graph_summary(conditions(c)))


def test_historical_motion_and_actual_timestamps_are_preserved():
    a=graph(times=3);a['features'][0,:,1,4]=[0,1,2]
    b=graph(times=3);b['features'][0,:,1,4]=[1,0,2]
    pa=production_graph_summary(conditions(a),historical=True,history_timestamps=[1.,2.],timestamp=3.)
    pb=production_graph_summary(conditions(b),historical=True,history_timestamps=[1.,2.],timestamp=3.)
    assert not np.array_equal(pa,pb)
    pc=production_graph_summary(conditions(a),historical=True,history_timestamps=[.5,2.],timestamp=3.)
    assert not np.array_equal(pa,pc)
    with pytest.raises(ValueError):production_graph_summary(conditions(a),historical=True,history_timestamps=[1.,4.],timestamp=3.)


def test_future_actions_and_unauthenticated_physical_label_bits_do_not_change_context():
    class Norm:
        def transform(self,x):return x
    actions=np.zeros((96,23),np.float32);changed=actions.copy();changed[32:]=9
    a=past_action_summary(actions,32,Norm());b=past_action_summary(changed,32,Norm())
    assert np.array_equal(a[0],b[0]) and a[1]==b[1]
    g=graph();baseline=production_graph_summary(conditions(g));g['features'][...,23:30]=19
    assert np.array_equal(baseline,production_graph_summary(conditions(g)))
    mask=after_feature_mask({},conditions(g),cam_rel_poses=None)
    assert mask.shape==(1518,) and not mask[279:318].any() and not mask[357:378].any()


def library(tmp_path):
    directory=tmp_path/'lib';directory.mkdir()
    contexts=np.asarray([[0,0,0,0,0],[0,2,0,0,0],[0,0,2,0,0],[0,0,0,0,0]],np.float32)
    arrays=dict(context=contexts,actual_native=np.zeros((4,32,23),np.float32),action_mask=np.ones((4,32),bool),
        reference_state=np.zeros((4,32),np.float32),reference_after_state=np.zeros((4,32),np.float32),
        reference_visual_context=np.zeros((4,1),np.float32),after_features=np.zeros((4,2),np.float32),
        after_feature_mask=np.ones((4,2),bool),next_index=np.zeros(4,np.int64),next_mask=np.ones(4,bool))
    np.savez(directory/'samples.npz',**arrays);np.savez(directory/'codec.npz',zero=np.zeros(1))
    rows=[dict(episode_id=str(i),split='validation' if i==3 else 'train',sequence_steps=[32,64],before_timestamp=32/30,after_timestamp=64/30) for i in range(4)]
    (directory/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    md=dict(context_width=5,artifact_sha256={n:h(directory/n) for n in ('samples.npz','codec.npz','rows.jsonl')},
        retrieval_distance_groups=[dict(name='visual',slices=[[0,1]],weight=.2),dict(name='robot',slices=[[1,2]],weight=.25),
            dict(name='graph',slices=[[2,3]],weight=.3),dict(name='history',slices=[[3,5]],weight=.25)])
    (directory/'identity.json').write_text(json.dumps(md))
    return RetrievalDataset(directory)


def test_state_and_target_graph_change_nearest_reference_and_episode_is_excluded(tmp_path):
    data=library(tmp_path);query=np.zeros(5,np.float32)
    assert data.query(query)[0][0]['episode_id']=='0'
    query[1]=2;assert data.query(query)[0][0]['episode_id']=='1'
    query[1]=0;query[2]=2;assert data.query(query)[0][0]['episode_id']=='2'
    query[:]=0;assert data.query(query,exclude_episode_id='0')[0][0]['episode_id']!='0'
    distances,pieces=data.distance_components(query,data.arrays['context'][:3])
    assert set(pieces)=={'visual','robot','graph','history'}
    assert distances[1]==pytest.approx(.25*4) and distances[2]==pytest.approx(.3*4)


def test_changed_artifact_is_rejected(tmp_path):
    data=library(tmp_path)
    with (data.directory/'rows.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='artifact changed'):RetrievalDataset(data.directory)
