

from symmetry_no.utilities3 import *
import matplotlib.pyplot as plt
import numpy as np

PATH = '/media/wumming/HHD/HHD_data/'
TRAIN_PATH = PATH + 'piececonst_r421_N1024_smooth1.mat'

reader = MatReader(TRAIN_PATH)
adata = reader.read_field('coeff')
udata = reader.read_field('sol')

for s in range(6,20):
    r = int(420/s)
    # s = int(420/r)+1
    a = adata[s, ::r, ::r][:s, :s]
    u = udata[s, ::r, ::r][:s, :s]


    # Create some data: a 2D grid of values
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y)

    # Create the plot
    # plt.figure(figsize=(6, 6))
    # c = plt.pcolormesh(X, Y, a, cmap='viridis', shading='auto', edgecolors='k', linewidths=2)
    # plt.axis('off')
    # plt.savefig(f'/home/wumming/Documents/symmetry-no/data/res_imgs/input_{s}.svg', bbox_inches='tight', pad_inches=0)

    plt.figure(figsize=(6, 6))
    c = plt.pcolormesh(X, Y, u, cmap='viridis', shading='auto', edgecolors='k', linewidths=2)
    plt.axis('off')
    plt.savefig(f'/home/wumming/Documents/symmetry-no/data/res_imgs/output_{s}.pdf', bbox_inches='tight', pad_inches=0)
