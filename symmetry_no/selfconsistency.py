import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from symmetry_no.data_augmentation import RandomCropResize
from symmetry_no.darcy_utilities import DarcyExtractBC
from symmetry_no.helmholtz_utilities import HelmholtzExtractBC


def LossSelfconsistency(model,x,loss_fn,y=None,re=None,rate=None,type="darcy",plot=False):
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

    if type == "darcy":
        ExtractBD = DarcyExtractBC
    elif type == "helmholtz":
        ExtractBD = HelmholtzExtractBC
    elif type == "NS":
        ExtractBD = lambda x, y : x # No boundary
    else:
        print("boundary type not supported")

    if re == None:
        re = torch.ones(batch_size, 1, requires_grad=False).to(x.device)

    # If y is given, we use it as the ground truth. the gradient only flow to y_small_
    if y!=None:
        # resample on smaller domain
        i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)
        #
        x_small = transform_xy.crop(x, i, j, h, w)
        y_small = transform_xy.crop(y, i, j, h, w)

        x_small = ExtractBD(x_small, y_small)
        #
        y_small_ = model(x_small, re)
        return loss_fn(y_small_, y_small)

    # If y is not given, we set y=model(x). We treat the subdomain as ground truth can detach y_small_
    else:
        y = model(x, re)

        # resample on smaller domain
        i,j,h,w,re = transform_xy.get_params(x, y, re=re, rate=rate)
        #
        x_small = transform_xy.crop(x,i,j,h,w)
        y_small = transform_xy.crop(y,i,j,h,w)

        x_small = ExtractBD(x_small,y_small)
        #
        y_small_ = model(x_small, re)
        # return loss_fn(y_small_.view(batch_size,-1).detach(), y_small.view(batch_size,-1))


        # ### PLOT
        if plot:
            x_l = x[0,0].squeeze().detach().cpu().numpy()
            x_s = x_small[0,0].squeeze().detach().cpu().numpy()
            g_l = x[0,1].squeeze().detach().cpu().numpy()
            g_s = x_small[0, 1].squeeze().detach().cpu().numpy()
            y_l = y[0].squeeze().detach().cpu().numpy()
            y_s = y_small_[0].squeeze().detach().cpu().numpy()

            fig, axs = plt.subplots(2, 3)  # Creates a grid of 2x2 for the images

            # Plot each image in its respective subplot
            im = axs[0, 0].imshow(x_l, cmap='viridis')
            fig.colorbar(im, ax=axs[0, 0], orientation='vertical')
            im = axs[0, 1].imshow(g_l, cmap='viridis')
            fig.colorbar(im, ax=axs[0, 1], orientation='vertical')
            im = axs[0, 2].imshow(y_l, cmap='viridis')
            fig.colorbar(im, ax=axs[0, 2], orientation='vertical')

            im = axs[1, 0].imshow(x_s, cmap='viridis')
            fig.colorbar(im, ax=axs[1, 0], orientation='vertical')
            im = axs[1, 1].imshow(g_s, cmap='viridis')
            fig.colorbar(im, ax=axs[1, 1], orientation='vertical')
            im = axs[1, 2].imshow(y_s, cmap='viridis')
            fig.colorbar(im, ax=axs[1, 2], orientation='vertical')

            # fig.show()
            plt.savefig(f'darcy_sc.png')
            print("finish plotting")

        return loss_fn(y_small, y_small_.detach())

