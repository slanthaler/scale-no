import matplotlib.pyplot as plt
import torch

def FD_x(a,dx):
    """
    Finite difference in x-direction.
    """
    return (a[...,1:,:] - a[...,:-1,:])/dx

def FD_y(a,dy):
    """
    Finite difference in y-direction.
    """
    return (a[...,:,1:] - a[...,:,:-1])/dy

def AV_x(a):
    """
    Averaging in x-direction.
    """
    return (a[...,1:,:] + a[...,:-1,:])/2.

def AV_y(a):
    """
    Averaging in y-direction.
    """
    return (a[...,:,1:] + a[...,:,:-1])/2.

def PDEResidual(x,y):
    """
    Compute the pointwise PDE residual: residual = d(a*du)
    """
    # filter out the data etc.
    u = y[...,0,:,:]
    a = x[...,0,:,:]
    Nx = a.shape[-1]
    dx = 1/Nx
    
    # 
    u_x = FD_x(u,dx)
    u_y = FD_y(u,dx)
    au_xx = FD_x(AV_x(a) * FD_x(u,dx), dx)
    au_yy = FD_y(AV_y(a) * FD_y(u,dx), dx)
    #
    au_xx = au_xx[...,:,1:-1]
    au_yy = au_yy[...,1:-1,:]

    #
    return au_xx+au_yy


def PlotResidual(x,y,subsamp=8):
    """
    Plot pointwise residual (Darcy PDE).
    """
    res = PDEResidual(x,y)
    
    if len(x.shape)<4:
        NotImplementedError(f'PlotResidual: {x.shape=} is not supported. Need 4 indices!')

    for i in range(x.shape[0]):
        a = x[i,0,::subsamp,::subsamp].squeeze().detach().numpy()
        u = y[i,0,::subsamp,::subsamp].squeeze().detach().numpy()
        resi = res[i,::subsamp,::subsamp].squeeze().detach().numpy()
        #
        fig, axs = plt.subplots(1,3,figsize=(10,3))
    
        # plot a
        pc0 = axs[0].pcolor(a.T)
        fig.colorbar(pc0,ax=axs[0])
        axs[0].set_title('a(x)')
    
        # plot u 
        pc1 = axs[1].pcolor(u.T)
        fig.colorbar(pc1,ax=axs[1])
        axs[1].set_title('u(x)')
    
        # plot residual
        pc2 = axs[2].pcolor(resi.T)
        fig.colorbar(pc2,ax=axs[2])
        axs[2].set_title('residual(x)')



def DarcyExtractBC(x,y):
    """
    Extract boundary conditions from y and add them to channels 1,...,4 in x.
    """
    unsqueeze_x = (x.ndim==3) # check whether batch dimension is missing
    unsqueeze_y = (y.ndim==3)

    # add batch dimension
    if unsqueeze_x:
        x = x.unsqueeze(0)
    if unsqueeze_y:
        y = y.unsqueeze(0)

    # extract BC
    Nx, Ny = y.shape[-2], y.shape[-1]
    x[:,1,:,:] = y[:,0, 0, :].unsqueeze(1).repeat(1,Nx,1)  # left boundary
    x[:,2,:,:] = y[:,0, :, 0].unsqueeze(2).repeat(1,1,Ny)  # lower boundary
    x[:,3,:,:] = y[:,0,-1, :].unsqueeze(1).repeat(1,Nx,1)  # right boundary
    x[:,4,:,:] = y[:,0, :,-1].unsqueeze(2).repeat(1,1,Ny)  # upper boundary

    # undo adding of batch dimension
    if unsqueeze_x:
        x = x[0]
    if unsqueeze_y:
        y = y[0]

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
