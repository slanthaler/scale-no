import torch
import numpy as np

Re = 10000
sub = 1

Path = '/media/wumming/HHD/HHD_data/KF/'
# Path = '/central/groups/tensorlab/zongyi/KF/'

# data = torch.load(Path+'KF_f0_Re'+str(Re)+'_N25_T500_S256_part0.pt')
data = torch.load(Path+'KF_f0_Re'+str(Re)+'_N10_T300_S1024_part0.pt')
print(data.shape)

S = data.shape[-1] // sub
new_data = data[-5:,:301,::sub, ::sub].clone()

print(S, new_data.shape)
torch.save(new_data, Path+'KF_f0_Re'+str(Re)+'_N5_T300_S'+str(S)+'_test.pt')


