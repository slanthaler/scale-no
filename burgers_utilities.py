import matplotlib.pyplot as plt
import torch



def BurgersExtractBC(y):
    """
    Extract boundary conditions from y and add them to channels 1,...,4 in x.
    x (batch, T_in, x)
    y (batch, T_out, x)
    """

    if y.ndim > 3:
        y = y.squeeze()
    T, S = y.shape[1], y.shape[2]


    # extract BC
    boundary0 = y[..., 0]
    boundary1 = y[..., -1]
    boundary = torch.stack([boundary0, boundary1], dim=1) # (batch, 2, T)
    boundary = boundary.unsqueeze(-1).repeat(1,1,1,S) # (batch, 2, T, S)

    x = y[:, 0, :]  # x (batch, x)
    x = x.reshape(-1, 1, 1, S).repeat(1, 1, T, 1)  # x (batch, 1, T, x)
    x = torch.cat([x, boundary], dim=1)  # x (batch, 1+2, T, x)
    return x
        
def Upsample(x, size):
    """
    given input x on [d,d], we double the domain to [s,s]
    """
    while x.shape[-1] < size:
        x = torch.cat([x, x.flip(-2)], dim=-2)
        x = torch.cat([x, x.flip(-1)], dim=-1)
    x = x[..., :size, :size]
    return x
