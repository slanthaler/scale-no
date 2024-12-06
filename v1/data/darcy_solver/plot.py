# Script to produce plots illustrating a Darcy flow dataset

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse

# add path to symmetryno package
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#
from symmetry_no.data_loader import DarcyReader

filename =  '/media/wumming/HHD/HHD_data/darcy/darcy_test_BC_tanh_amin1.0_amax12.0_Nsamp256_Ns128_alphacoeff1.50_alphag3.50.mat'

# load training and test datasets
print('Loading datasets...')
data = DarcyReader(filename, root_dir="")

# plot some samples
print('Plotting samples...')

# get a random index set
idx_set = np.random.choice(data.x.shape[0], 4, replace=False)
x_tags = ['Pressure', 'BC1', 'BC2', 'BC3', 'BC4']

savename = "Ns128"

for i in idx_set:
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.subplots_adjust(hspace = .1, wspace=.001)
    axs = axs.ravel()

    for j in range(9):
        axs[j].axis('off')

    subplot_idx = [0,1,3,7,5]
    clim = [np.min(data.y[i].numpy()), np.max(data.y[i].numpy())]

    # subplot for pressure field
    si = 0
    im = axs[si].imshow(data.x[i][0], cmap='viridis')
    axs[si].set_title(f'{x_tags[0]}')
    axs[si].axis('off')
    fig.colorbar(im, ax=axs[si])

    # subplots for x
    for j in range(1,5):
        si = subplot_idx[j]
        im = axs[si].imshow(data.x[i][j], cmap='viridis', clim=clim)
        axs[si].set_title(f'{x_tags[j]}')
        axs[si].axis('off')
        fig.colorbar(im, ax=axs[si])

    # plot for y
    si = 4
    im = axs[si].imshow(data.y[i][0], cmap='viridis', clim=clim)
    axs[si].set_title(f'Solution')
    axs[si].axis('off')
    fig.colorbar(im, ax=axs[si])

    # save plot
    plt.savefig(f'samples_{i}_{savename}.png')
    # fig.show()
