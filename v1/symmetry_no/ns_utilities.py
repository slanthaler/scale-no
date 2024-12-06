import torch
import numpy as np

def NSExtractBC(y):
    """
    In (B, T, x, y)
    Out (B, 5, x, y, T)
    [base, top, bottom, left, right]
    """
    unsqueeze_y = (y.ndim == 3)

    # add batch dimension
    if unsqueeze_y:
        y = y.unsqueeze(0)

    batch_size, T, S_x, S_y = y.shape

    base = y[:, 0, :, :]
    top = y[..., 0, :]
    bottom = y[..., -1, :]
    left = y[..., :, 0]
    right = y[..., :, -1]
    
    base = base.unsqueeze(1).expand(-1, T, -1, -1)
    top = top.unsqueeze(2).expand(-1, -1, S_x, -1)
    bottom = bottom.unsqueeze(2).expand(-1, -1, S_x, -1)
    left = left.unsqueeze(3).expand(-1, -1, -1, S_y)
    right = right.unsqueeze(3).expand(-1, -1, -1, S_y)

    x = torch.stack([base, top, bottom, left, right], dim=1)

    # undo adding of batch dimension
    if unsqueeze_y:
        x = x[0]

    return x

def Upsample(x, size):
    """
    given input x on [d,d], we double the domain to [s,s]
    """
    while x.shape[-2] < size:
        x = torch.cat([x, x.flip(-3)], dim=-3)
        x = torch.cat([x, x.flip(-2)], dim=-2)
    x = x[..., :size, :size, :]
    return x
