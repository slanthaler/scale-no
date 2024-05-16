import torch
import numpy as np

Re = 1000
Path = '/media/wumming/HHD/HHD_data/KF/'
# Path = '/central/groups/tensorlab/zongyi/KF/'

data0 = torch.load(Path+'KF_f0_Re'+str(Re)+'_N25_T500_S256_part0.pt')
data1 = torch.load(Path+'KF_f0_Re'+str(Re)+'_N25_T500_S256_part1.pt')
data2 = torch.load(Path+'KF_f0_Re'+str(Re)+'_N25_T500_S256_part2.pt')

print(data0.shape)
data = torch.cat([data0, data1, data2], dim=0)[:60, :301].clone()

print(data.shape)
torch.save(data, Path+'KF_f0_Re'+str(Re)+'_N60_T300_S256_train.pt')


