import torch
import numpy as np

data = torch.load('/media/wumming/HHD/HHD_data/KF/KF_f0_Re250_N25_T500_S256_part0.pt')
# data = torch.load('/media/wumming/HHD/HHD_data/KF/KF_f0_Re4000_N10_T300_S1024_dt_adpt_dx_2048_part0.pt')
print(data.shape)
sub = 4
data = data[:,:301,::sub, ::sub]
torch.save(data, '/media/wumming/HHD/HHD_data/KF/KF_f0_Re250_N25_T300_S64_part0.pt')
# data = torch.load('/media/wumming/HHD/HHD_data/KF/KF_f0_Re4000_N10_T300_S1024_dt_adpt_dx_2048_part0.pt')
