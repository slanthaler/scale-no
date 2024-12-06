import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from symmetry_no.data_augmentation import RandomCropResize, RandomCropResizeTime, RandomCropResizeTimeAR
from symmetry_no.darcy_utilities import DarcyExtractBC
from symmetry_no.helmholtz_utilities import HelmholtzExtractBC
from symmetry_no.burgers_utilities import BurgersExtractBC

import time

def LossSelfconsistency(model,x,loss_fn,y=None,re=None,rate=None,new_y=None,size_min=32,type="darcy",plot=False,group_action=None,align_corner=False):
    """
    Selfconsistency loss:

    Enforces that the model evaluated on the
    entire domain, and restricted to a subdomain
    be equal to the model directly evaluated on
    the subdomain.

    The subdomain is chosen randomly each time.
    """
    #

    batch_size = x.shape[0]

    if type == "darcy":
        ExtractBD = DarcyExtractBC
        transform_xy = RandomCropResize(p=1.0, size_min=size_min)
    elif type == "helmholtz":
        ExtractBD = HelmholtzExtractBC
        transform_xy = RandomCropResize(p=1.0, size_min=size_min)
    elif type == "NS":
        ExtractBD = lambda x, y : x # No boundary
        transform_xy = RandomCropResizeTime(p=1.0, size_min=size_min)  # Changed from RandomCropResizeTimeAR to RandomCropResizeTime
    elif type == "burgers":
        ExtractBD = lambda x, y : BurgersExtractBC(y)
        transform_xy = RandomCropResizeTime(p=1.0, size_min=size_min)
    else:
        print("boundary type not supported")

    if re == None:
        re = torch.ones(batch_size, 1, requires_grad=False).to(x.device)

    # If y is given, we use it as the ground truth. the gradient only flow to y_small_
    if y is not None:
        # resample on smaller domain
        if isinstance(transform_xy, RandomCropResizeTime):
            i, j, k, h, w, t, re = transform_xy.get_params(x, y, re=re, rate=rate)
            x_small = transform_xy.crop(x, i, j, k, h, w, t)
            y_small = transform_xy.crop(y, i, j, k, h, w, t)
        else:
            i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)
            x_small = transform_xy.crop(x, i, j, h, w)
            y_small = transform_xy.crop(y, i, j, h, w)

        x_small = ExtractBD(x_small, y_small)

        if group_action is not None:
            x_small, y_small = group_action(x_small, y_small)
        y_small_ = model(x_small, re)

        return loss_fn(y_small_, y_small)

    # If y is not given, we set y=model(x). We treat the subdomain as ground truth can detach y_small_
    else:
        mode = "sc"

        y = model(x, re)

        # resample on smaller domain
        if isinstance(transform_xy, RandomCropResizeTime):
            i, j, k, h, w, t, re = transform_xy.get_params(x, y, re=re, rate=rate)
            if align_corner:
                i = j = k = 0
            x_small = transform_xy.crop(x, i, j, k, h, w, t)
            y_small = transform_xy.crop(y, i, j, k, h, w, t)
        else:
            i, j, h, w, re = transform_xy.get_params(x, y, re=re, rate=rate)
            if align_corner:
                i = j = 0
            x_small = transform_xy.crop(x, i, j, h, w)
            y_small = transform_xy.crop(y, i, j, h, w)

        x_small = ExtractBD(x_small, y_small)
        y_small_ = model(x_small.detach(), re)

        if align_corner:
            if isinstance(transform_xy, RandomCropResizeTime):
                H, W, T = h//2, w//2, t//2
                y_small = y_small[..., :H, :W, :T]
                y_small_ = y_small_[..., :H, :W, :T]
            else:
                H, W = h//2, w//2
                y_small = y_small[..., :H, :W]
                y_small_ = y_small_[..., :H, :W]

        return loss_fn.truncated(y_small_, y_small)

# ### PLOT
def plot_boundary(x, x_small, y, y_small_):
    x_l = x[0,0].squeeze().detach().cpu().numpy()
    x_s = x_small[0,0].squeeze().detach().cpu().numpy()
    g_l = x[0,1].squeeze().detach().cpu().numpy()
    g_l[g_l.shape[0]//8:] = np.nan
    g_s = x_small[0,1].squeeze().detach().cpu().numpy()
    g_s[g_s.shape[0]//8:] = np.nan
    y_l = y[0,0].squeeze().detach().cpu().numpy()
    y_s = y_small_[0,0].squeeze().detach().cpu().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Creates a grid of 2x2 for the images

    # Plot each image in its respective subplot
    im = axs[0, 0].imshow(x_l, cmap='viridis', vmin=np.min(x_l), vmax=np.max(x_l))
    # fig.colorbar(im, ax=axs[0, 0], orientation='vertical')
    axs[0, 0].set_title('input coeff')
    im = axs[0, 1].imshow(g_l, cmap='viridis', vmin=np.min(y_l), vmax=np.max(y_l))
    # fig.colorbar(im, ax=axs[0, 1], orientation='vertical')
    axs[0, 1].set_title('input boundary')
    im = axs[0, 2].imshow(y_l, cmap='viridis', vmin=np.min(y_l), vmax=np.max(y_l))
    # fig.colorbar(im, ax=axs[0, 2], orientation='vertical')
    axs[0, 2].set_title('output')

    im = axs[1, 0].imshow(x_s, cmap='viridis', vmin=np.min(x_l), vmax=np.max(x_l))
    # fig.colorbar(im, ax=axs[1, 0], orientation='vertical')
    axs[1, 0].set_title('subdomain input coeff')
    im = axs[1, 1].imshow(g_s, cmap='viridis', vmin=np.min(y_l), vmax=np.max(y_l))
    # fig.colorbar(im, ax=axs[1, 1], orientation='vertical')
    axs[1, 1].set_title('subdomain input boundary')
    im = axs[1, 2].imshow(y_s, cmap='viridis', vmin=np.min(y_l), vmax=np.max(y_l))
    # fig.colorbar(im, ax=axs[1, 2], orientation='vertical')
    axs[1, 2].set_title('subdomain output')

    # fig.show()
    plt.tight_layout()
    plt.savefig(f'helm_sc.png')
    print("finish plotting 1")

    x = x[0].squeeze().detach().cpu().numpy()
    x_s = x_small[0].squeeze().detach().cpu().numpy()
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))  # Creates a grid of 2x2 for the images
    im = axs[0, 0].plot(y_l[0, :], label="output boundary")
    im = axs[0, 0].plot(x[1, 0, :], label="input boundary")
    axs[0, 0].legend()
    im = axs[1, 0].plot(y_s[0, :], label="subdomain output boundary")
    im = axs[1, 0].plot(x_s[1, 0, :], label="subdomain input boundary")
    axs[0, 0].set_title('top boundary')
    axs[1, 0].legend()

    im = axs[0, 1].plot(y_l[:, 0], label="output boundary")
    im = axs[0, 1].plot(x[3, :, 0], label="input boundary")
    axs[0, 1].legend()
    im = axs[1, 1].plot(y_s[:, 0], label="subdomain output boundary")
    im = axs[1, 1].plot(x_s[3, :, 0], label="subdomain input boundary")
    axs[0, 1].set_title('bottom boundary')
    axs[1, 1].legend()

    im = axs[0, 2].plot(y_l[-1, :], label="output boundary")
    im = axs[0, 2].plot(x[5, -1, :], label="input boundary")
    axs[0, 2].legend()
    im = axs[1, 2].plot(y_s[-1, :], label="subdomain output boundary")
    im = axs[1, 2].plot(x_s[5, -1, :], label="subdomain input boundary")
    axs[0, 2].set_title('left boundary')
    axs[1, 2].legend()

    im = axs[0, 3].plot(y_l[:, -1], label="output boundary")
    im = axs[0, 3].plot(x[7, :, -1], label="input boundary")
    axs[0, 3].legend()
    im = axs[1, 3].plot(y_s[:, -1], label="subdomain output boundary")
    im = axs[1, 3].plot(x_s[7, :, -1], label="subdomain input boundary")
    axs[0, 3].set_title('right boundary')
    axs[1, 3].legend()

    plt.tight_layout()
    plt.savefig(f'helm_boundary.png')
    print("finish plotting 2")

    time.sleep(1000)
