# Script to produce plots illustrating a Darcy flow dataset

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# add path to symmetryno package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#
from symmetry_no.data_loader import DarcyReader

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', "--filename",
                    type=str,
                    help="Specify filename of dataset.",
                    required=True)
parser.add_argument('-d', "--folder",
                    type=str,
                    help="Specify folder to save plots.",
                    required=True)
parser.add_argument('-p', "--plots",
                    type=int,
                    help="Specify number of plots to produce.",
                    default=10)
args = parser.parse_args()

# load training and test datasets
print('Loading datasets...')
data = DarcyReader(args.filename)

# create folder for plots
os.mkdir(args.folder) if not os.path.exists(args.folder) else None

# filename without suffix
filename = os.path.basename(args.filename)

# plot some samples
print('Plotting samples...')

# get a random index set
idx_set = np.array([])
while len(idx_set) < args.plots:
    idx_set = np.unique(np.random.randint(0, data.x.shape[0], 10*args.plots))[0:args.plots]
x_tags = ['Pressure', 'BC1', 'BC2', 'BC3', 'BC4']

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
    plt.savefig(os.path.join(args.folder, f'samples_{i}_{filename}.png'))