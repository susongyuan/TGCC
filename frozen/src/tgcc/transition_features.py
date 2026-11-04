"""Versioned causal execution changes; availability is not physical progress."""
import numpy as np

FEATURE_SCHEMA = 'tgcc.support_features.v3.delta_stable_motion_no_metadata'
EXECUTION_TARGET = 'actual_after_minus_actual_before'


def quaternion_slices(side):
    return [(side+28,side+32),(side+35,side+39),
            *[(side+78+7*k+3,side+78+7*(k+1)) for k in range(3)]]


def canonical_quaternion(q):
    q=np.asarray(q,np.float32)
    normalized=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
    major=np.take_along_axis(normalized,np.abs(normalized).argmax(-1)[...,None],axis=-1)
    return normalized*np.where(major<0,-1.,1.)


def stable_scoring_context(context, visual_width=256):
    """Keep the native policy input unchanged; score/index stable motion facts.

    Raw instantaneous velocity channels are near-constant in the recorded
    demonstrations and fluctuate under deployed position control. Causal
    graph pose differences and past commands retain measured motion instead.
    """
    x=np.array(context,dtype=np.float32,copy=True);v=visual_width
    x[...,v:v+3]=0
    side=v+70
    x[...,side+3:side+25]=0
    for lo,hi in quaternion_slices(side):x[...,lo:hi]=canonical_quaternion(x[...,lo:hi])
    # Apply the same transform to legacy real raw caches and newly extracted
    # graphs, including their weighted aggregate confidence. No Oracle
    # knownness may survive through the historical task graph either.
    for base in (v+211+366,v+1309+495):
        for role in (9,10):x[...,base+15*role]=0
        x[...,base+364]=(x[...,base:base+180:15]*x[...,base+222:base+234]).sum(-1)
    history=v+2794
    x[...,history:history+3]=0;x[...,history+23:history+26]=0
    return x


def current_outcome_from_context(context, visual_width=256):
    """Read current facts only; skip previous actions and temporal summaries."""
    x = np.asarray(context, np.float32)
    v = visual_width
    return np.concatenate((x[..., :v+23], x[..., v+70:v+1309]), axis=-1)


def outcome_mask(values, visual_width=256):
    """Observed values in the raw current-outcome representation.

    Validity bits belong to the key, but are not execution targets. UNKNOWN
    at either end of a transition must never become a measured change.
    """
    x = np.asarray(values)
    if x.shape[-1] != visual_width + 1262:
        raise ValueError('Execution fact layout mismatch')
    mask = np.ones(x.shape, bool)
    side = visual_width + 23
    mask[..., side:side+39] = x[..., side+39:side+78] > .5
    mask[..., side+78:side+99] = x[..., side+99:side+120] > .5
    mask[..., side+123:side+132] = x[..., side+132:side+141] > .5
    mask[...,visual_width:visual_width+3]=False
    mask[...,side+3:side+25]=False
    for lo, hi in ((39,78), (99,123), (132,141)):
        mask[..., side+lo:side+hi] = False
    for lo,hi in quaternion_slices(side):
        mask[...,lo:hi] &= (np.linalg.norm(x[...,lo:hi],axis=-1)>1e-6)[...,None]
    graphstart = side + 141
    for layer in range(3):
        base = graphstart + layer*366
        graph = x[..., base:base+366]
        for role in range(12):
            present = graph[..., 222+role] != 0
            mask[..., base+role*15:base+(role+1)*15] &= present[..., None]
            mask[..., base+198+role*2:base+200+role*2] &= present[..., None]
            # Confidence, staleness and extent-availability are metadata.
            mask[..., base+role*15] = False
            mask[..., base+role*15+1] = False
            mask[..., base+role*15+14] = False
            extent_known = graph[..., role*15+14] != 0
            mask[..., base+role*15+11:base+role*15+14] &= extent_known[..., None]
        relations_known = np.any(graph[..., 234:266] != 0, axis=-1)
        mask[..., base+234:base+362] &= relations_known[..., None]
        mask[..., base+362:base+366] = False
        if layer == 1:
            for role in range(12):
                mask[..., base+role*15+11:base+role*15+15] = False
    return mask


def transition_target(before, after, before_mask, after_mask):
    before, after = np.asarray(before,np.float32), np.asarray(after,np.float32)
    valid = np.asarray(before_mask,bool) & np.asarray(after_mask,bool)
    if before.shape != after.shape or valid.shape != before.shape:
        raise ValueError('Causal execution pair/mask mismatch')
    if not np.isfinite(before[valid]).all() or not np.isfinite(after[valid]).all():
        raise ValueError('Nonfinite observed execution change')
    if before.shape[-1]>=1262:
        before,after=before.copy(),after.copy()
        side=before.shape[-1]-1262+23
        for lo,hi in quaternion_slices(side):
            q=canonical_quaternion(before[...,lo:hi])
            other=after[...,lo:hi]/np.maximum(np.linalg.norm(after[...,lo:hi],axis=-1,keepdims=True),1e-8)
            other*=np.where((q*other).sum(-1,keepdims=True)<0,-1.,1.)
            before[...,lo:hi],after[...,lo:hi]=q,other
    return np.where(valid, after-before, 0).astype(np.float32), valid


def fit_transition_statistics(targets, masks, train_indices):
    x, m = np.asarray(targets,np.float64)[train_indices], np.asarray(masks,bool)[train_indices]
    if x.shape != m.shape or not len(x) or not np.isfinite(x[m]).all():
        raise ValueError('Invalid observed training transition statistics')
    count = m.sum(0)
    mean = (np.where(m,x,0).sum(0) / np.maximum(count,1))
    variance = np.where(m,(x-mean)**2,0).sum(0) / np.maximum(count,1)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance),.01).astype(np.float32), count > 0


def normalize_transition(change, known, artifact):
    """One train-supported target space for library, training and deployment."""
    change, known = np.asarray(change,np.float32), np.asarray(known,bool)
    supported = np.asarray(artifact['after_supported'],bool)
    mean, std = artifact['after_mean'], artifact['after_std']
    if change.shape != known.shape or change.shape[-1:] != supported.shape:
        raise ValueError('Transition normalization shape mismatch')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid frozen transition statistics')
    valid = known & supported
    if not np.isfinite(change[valid]).all():
        raise ValueError('Nonfinite supported execution change')
    return np.where(valid,(change-mean)/std,0).astype(np.float32),valid
