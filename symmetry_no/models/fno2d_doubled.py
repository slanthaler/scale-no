"""

Taken from
[https://github.com/neuraloperator/neuraloperator/blob/master/fourier_2d.py]
@author: Zongyi Li (with some minor tweaks by Sam Lanthaler)

as discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).

"""

import torch.nn.functional as F
from timeit import default_timer
from symmetry_no.utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d_doubled(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_doubled, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize, C, Nx, Ny = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        sub = 4
        x = torch.cat([x, -x.flip(dims=[-2])[..., 1:Nx-1:sub, :]], dim=-2)
        x = torch.cat([x, -x.flip(dims=[-1])[..., :, 1:Ny-1:sub]], dim=-1)
        x_ft = torch.fft.rfft2(x)

        m1 = np.minimum(self.modes1, Nx//2)
        m2 = np.minimum(self.modes2, Ny//2)


        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[..., :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[..., -m1:, :m2])

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        x = x[:,:,:Nx,:Ny]
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d_doubled(nn.Module):
    def __init__(self, modes1, modes2, width, depth=4, in_channel=7, out_channel=1):
        super(FNO2d_doubled, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function, boundary conditions and locations (a(x, y), BC1, BC2, BC3, BC4, x, y)
        input shape: (batchsize, x=s, y=s, c=7)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel

        #self.p = nn.Linear(self.in_channel, self.width) # input channel is 7: (a(x, y), BC_left, BC_bottom, BC_right, BC_top, x, y)
        self.p = MLP(self.in_channel, self.width, self.width)

        self.conv = []
        self.mlp = []
        self.w = []
        for _ in range(depth):
            self.conv.append(SpectralConv2d_doubled(self.width, self.width, self.modes1, self.modes2))
            self.mlp.append(MLP(self.width, self.width, self.width))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
        #
        self.conv = nn.ModuleList(self.conv)
        self.mlp = nn.ModuleList(self.mlp)
        self.w = nn.ModuleList(self.w)
        #
        self.q = MLP(self.width, self.out_channel, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x, re=None):

        std = torch.std(x[:,1:].clone(), dim=[1,2,3], keepdim=True)
        x = torch.cat([x[:, :1], x[:, 1:] / std], dim=1)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1) # 1 is the channel dimension
        x = self.p(x)

        for i in range(self.depth):
            x1 = self.conv[i](x)
            x1 = self.mlp[i](x1)
            x2 = self.w[i](x)
            x = x1 + x2
            x = F.gelu(x)

        x = self.q(x)
        x = x*std
        del std
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
