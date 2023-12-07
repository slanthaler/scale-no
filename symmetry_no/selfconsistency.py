from symmetry_no.data_augmentation import RandomCropResize
from symmetry_no.darcy_utilities import DarcyExtractBC

def LossSelfconsistency(model,x,loss_fn):
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
    y = model(x)

    # resample on smaller domain
    i,j,h,w = transform_xy.get_params(x,y)
    #
    x_small = transform_xy.crop(x,i,j,h,w)
    y_small = transform_xy.crop(y,i,j,h,w)
    #
    x_small = DarcyExtractBC(x_small,y_small)
    
    # 
    y_small_ = model(x_small)
    return loss_fn(y_small_.view(batch_size,-1), y_small.view(batch_size,-1))
    
