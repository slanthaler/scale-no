import torch
import functorch as ft # for directional derivative computation

from symmetry_no.data_augmentation import RandomCropResize
from symmetry_no.darcy_utilities import DarcyExtractBC
from symmetry_no.finite_differences import FiniteDifferencer


def LossSelfconsistency(model,x,loss_fn,y=None):
    """
    Selfconsistency loss: 

    Enforces that the model evaluated on the 
    entire domain, and restricted to a subdomain 
    be equal to the model directly evaluated on 
    the subdomain. 

    The subdomain is chosen randomly each time.
    """
    #
    transform_xy = RandomCropResize(p=1.0)
    batch_size = x.shape[0]

    # If y is given, we use it as the ground truth. the gradient only flow to y_small_
    if y!=None:
        # resample on smaller domain
        i, j, h, w = transform_xy.get_params(x, y)
        #
        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)

        x_small = DarcyExtractBC(x_small, y_small)
        #
        y_small_ = model(x_small)
        return loss_fn(y_small_.view(batch_size, -1), y_small.view(batch_size, -1))

    # If y is not given, we set y=model(x). We treat the subdomain as ground truth can detach y_small_
    else:
        y = model(x)

        # resample on smaller domain
        i,j,h,w = transform_xy.get_params(x,y)
        #
        x_small = transform_xy.crop(x,i,j,h,w)
        y_small = transform_xy.crop(y,i,j,h,w)

        x_small = DarcyExtractBC(x_small,y_small)
        #
        y_small_ = model(x_small)
        return loss_fn(y_small_.view(batch_size,-1), y_small.view(batch_size,-1))
    

def LossSelfconsistencyDiff(model,x,loss_fn):
    """
    Selfconsistency loss: (differential version)

    Enforces self-consistency via infinitesimal symmetry relation:

    vG = DG/dcoeff*vcoeff + dG/dBC*vG, # this is in principle
       = DG/dx*vx                      # in practice, model=model(x)

    where:
        - G=model(coeff,BC)   -- model as function of coefficient and BC
        - vG = rDr(G)         -- (spatial) radial derivative of model output
        - vcoeff = rDr(coeff) -- (spatial) radial derivative of coefficient
        - vx = [vcoeff, ExtractBC(vG)]
    """
    # we note that radial derivative rDr is invariant under re-scaling of grid;
    # so we can choose grid on unit interval in both directions.
    # This **assumes** that the length of the underlying domain is the same in both directions!
    N0,N1 = x.shape[-2:]
    grid0,grid1 = torch.linspace(0,1,N0), torch.linspace(0,1,N1)
    FD = FiniteDifferencer(grid0,grid1)
    mid_pt = torch.rand((2,1,1)) # choose a random mid-point in [0,1]^2

    #
    G = model(x)                  # model prediction
    #
    vG = FD.rDr(G,mid_pt)         # radial derivative of model output
    coeff = x[:,0,:,:]            # coefficient field (get rid of BC)
    vcoeff = FD.rDr(coeff,mid_pt) # radial derivative of coefficient

    # extract BC
    vx = x.clone()
    vx[:,0,:,:] = vcoeff[:,:,:]   # coeff-component of vx
    DarcyExtractBC(vx,vG)         # extract BC of vG and insert in vx

    # compute directional derivative <dG/dx,vx> at input x
    _, dGdx_vx = ft.jvp(model, (x,), (vx,))

    return loss_fn(vG,dGdx_vx)

    
