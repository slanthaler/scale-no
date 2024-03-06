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
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.n_feature = 3
        in_channels = in_channels + 2*self.n_feature + self.n_feature # input + k_mat + Re
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # if self.modes1 == 0:
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, hidden_channels, dtype=torch.cfloat))
        self.bias1 = nn.Parameter(self.scale * torch.rand(1,hidden_channels,1,1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(hidden_channels, out_channels, dtype=torch.cfloat))
        self.bias2 = nn.Parameter(self.scale * torch.rand(1,out_channels,1,1, dtype=torch.cfloat))
        # else:
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        if weights.dim() == 4:
            return torch.einsum("bixy,ioxy->boxy", input, weights)
        if weights.dim() == 2:
            return torch.einsum("bixy,io->boxy", input, weights)

    def get_k(self, batchsize, S1, S2, device):
        # print("computing k")
        modes1, modes2 = S1 // 2, S2 // 2
        k_x1 = torch.cat((torch.arange(start=0, end=S1 - modes1, step=1, device=device),
                          torch.arange(start=-(modes1), end=0, step=1, device=device)), 0).\
                            reshape(S1, 1).repeat(1,S2).reshape(1, S1, S2)
        k_x2 = torch.cat((torch.arange(start=0, end=S2 - modes2, step=1, device=device),
                          torch.arange(start=-(modes2), end=0, step=1, device=device)), 0).\
                            reshape(1, S2).repeat(S1,1).reshape(1, S1, S2)

        k = torch.cat((k_x1, k_x2), 0).cfloat()[..., :modes2 + 1]
        k = k.reshape(1, 2, S1, modes2 + 1)

        # feature embedding
        pow = [i / (self.n_feature - 1) for i in range(self.n_feature)]
        k_mat = [k ** i for i in pow]
        k_mat = torch.cat(k_mat, dim=1).repeat(batchsize, 1,1,1)
        k_mat.requires_grad = False
        return k_mat

    def embed_re(self, S2, re):
        pow = [i / (self.n_feature - 1) for i in range(self.n_feature)]
        re_mat = [re ** i for i in pow]
        re_mat = torch.cat(re_mat, dim=1)[..., :S2//2 + 1]
        return re_mat

    def forward(self, x, Re):
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # add wavenumber k_mat
        k_mat = self.get_k(x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
        re_mat = self.embed_re(x.shape[-1], Re)
        x_ft = torch.cat([x_ft, k_mat, re_mat], dim=1)

        # if self.modes1 == 0:
            # MLP layer
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        out_ft = out_ft + self.bias1

        out_ft = F.gelu(out_ft.real) + 1j * F.gelu(out_ft.imag)
        # out_ft = torch.tanh(out_ft.abs()) * (out_ft / (out_ft.abs() + self.eps))

        out_ft = self.compl_mul2d(out_ft, self.weights2)
        out_ft1 = out_ft + self.bias2

        # else:
            # contraction layer
        m1 = np.minimum(self.modes1, x.size(-2)//2)
        m2 = np.minimum(self.modes2, x.size(-1)//2)
        out_ft2 = torch.zeros(x.size(0), self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft2[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights3[..., :m1, :m2])
        out_ft2[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights4[..., -m1:, :m2])

        out_ft = out_ft1 + out_ft2
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
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

class FNO_mlp(nn.Module):
    def __init__(self, width, modes1=0, modes2=0, depth=4, in_channel=7, out_channel=1):
        super(FNO_mlp, self).__init__()

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

        self.width = width
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel

        #self.p = nn.Linear(self.in_channel, self.width) # input channel is 7: (a(x, y), BC_left, BC_bottom, BC_right, BC_top, x, y)
        self.p = MLP(self.in_channel+1, self.width, self.width)

        self.conv = []
        self.mlp = []
        self.w = []
        for _ in range(depth):
            self.conv.append(SpectralConv2d(self.width, self.width, self.width, modes1, modes2))
            self.mlp.append(MLP(self.width, self.width, self.width))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
        #
        self.conv = nn.ModuleList(self.conv)
        self.mlp = nn.ModuleList(self.mlp)
        self.w = nn.ModuleList(self.w)
        #
        self.q = MLP(self.width, self.out_channel, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x, re=None):
        # x (batch, in_channels, X, Y)
        # re (batch, )

        std = torch.std(x[:,1:].clone(), dim=[1,2,3], keepdim=True)
        x = torch.cat([x[:, :1], x[:, 1:] / std], dim=1)

        grid = self.get_grid(x.shape, x.device)
        re = re.reshape(-1,1,1,1) * torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), requires_grad=False, device=x.device)
        x = torch.cat((x, grid, re), dim=1) # 1 is the channel dimension
        x = self.p(x)

        for i in range(self.depth):
            x1 = self.conv[i](x, re)
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
