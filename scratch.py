import torch
import numpy as np

for i in range(5):
    N = 2**(i+5)
    a = torch.linspace(0,1,N)
    a = torch.sin(2*np.pi*a)
    a_ft = torch.fft.fft(a, norm='forward')
    print(i, N, torch.norm(a_ft), torch.max(torch.abs(a_ft)))
    print(a_ft[:3])

