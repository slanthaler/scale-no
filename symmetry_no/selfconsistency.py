import torch
from symmetry_no.data_augmentation import RandomCropResize
from symmetry_no.darcy_utilities import DarcyExtractBC
from symmetry_no.helmholtz_utilities import HelmholtzExtractBC


def LossSelfconsistency(model,x,loss_fn,y=None,re=None,):
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

    if x.shape[1] == 5:
        ExtractBD = DarcyExtractBC
    elif x.shape[1] == 9:
        ExtractBD = HelmholtzExtractBC
    else:
        print("boundary type not supported")

    if re == None:
        re = torch.ones(batch_size, 1, requires_grad=False).to(x.device)

    # If y is given, we use it as the ground truth. the gradient only flow to y_small_
    if y!=None:
        # resample on smaller domain
        i, j, h, w, re = transform_xy.get_params(x, y, re=re)
        #
        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)

        x_small = ExtractBD(x_small, y_small)
        #
        y_small_ = model(x_small, re)
        return loss_fn(y_small_.view(batch_size, -1), y_small.view(batch_size, -1))

    # If y is not given, we set y=model(x). We treat the subdomain as ground truth can detach y_small_
    else:
        y = model(x, re)

        # resample on smaller domain
        i,j,h,w,re = transform_xy.get_params(x, y, re=re)
        #
        x_small = transform_xy.crop(x,i,j,h,w)
        y_small = transform_xy.crop(y,i,j,h,w)

        x_small = ExtractBD(x_small,y_small)
        #
        y_small_ = model(x_small, re)
        return loss_fn(y_small_.view(batch_size,-1).detach(), y_small.view(batch_size,-1))

