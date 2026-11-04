import torch


def make_att_2d_masks(pad_masks, att_masks):
    if pad_masks.ndim != 2 or att_masks.ndim != 2:
        raise ValueError("B,L masks required")
    blocks = att_masks.cumsum(1)
    return (
        (blocks[:, None, :] <= blocks[:, :, None])
        & pad_masks[:, None, :]
        & pad_masks[:, :, None]
    )
