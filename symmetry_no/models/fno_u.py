"""
@author: Zongyi Li and Catherine Deng

multiband FNO, with Re / k, mlp in channel expansion
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)
np.random.seed(0)

def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    if b.dim() == 4:
        return torch.einsum("bixy,ioxy->boxy", a, b)
    if b.dim() == 3:
        return torch.einsum("bixy,ixy->bixy", a, b)
    if b.dim() == 2:
        return torch.einsum("bixy,io->boxy", a, b)

# computes wavenumber matrix
class K:
    def __init__(self, ):
        self.n_feature = 3

    def compute_k(self, S1, S2, Re=1000):
        # print("computing k")
        modes1, modes2 = S1//2, S2//2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k_x1 = torch.cat((torch.arange(start=0, end=S1-modes1, step=1, device=device),
                        torch.arange(start=-(modes1), end=0, step=1, device=device)), 0).reshape(S1, 1).repeat(1, S2).reshape(1, S1, S2)
        k_x2 = torch.cat((torch.arange(start=0, end=S2-modes2, step=1, device=device),
                        torch.arange(start=-(modes2), end=0, step=1, device=device)), 0).reshape(1, S2).repeat(S1, 1).reshape(1, S1, S2)
        k = torch.cat((k_x1, k_x2), 0).cfloat()[..., :modes2+1]
        k = k.reshape(1, 2, S1, modes2+1)
        pow = [i/(self.n_feature-1) for i in range(self.n_feature)]
        k_mat = [k ** i for i in pow]
        k_mat = torch.cat(k_mat, dim=1)
        k_mat = k_mat * Re / 1000
        return k_mat

    def get_k(self, B, S1, S2):
        self.k = self.compute_k(S1, S2)
        k_ret = self.k.repeat(B, 1, 1, 1)
        return k_ret

    def get_Re(self, Re, S1, S2):
        Re = self.compute_k(S1, S2, Re)
        return Re

# f domain, multiplies modes by a weight matrix
class R_trans(nn.Module):
    def __init__(self, c_in, c_out, modes1, modes2, mlp=False): # TODO: double check this
        super(R_trans, self).__init__()

        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.mlp = mlp
        self.eps = 0.0001
        self.scale = (1 / (c_in * c_out))

        if self.mlp:
            self.weights1 = nn.Parameter(self.scale * torch.rand(self.c_in, self.c_out, dtype=torch.cfloat))
            self.bias1 = nn.Parameter(self.scale * torch.rand(self.c_out, 1, 1, dtype=torch.cfloat))
        else:
            self.weights1 = nn.Parameter(self.scale * torch.rand(self.c_in, self.c_out, self.modes1*2, self.modes2, dtype=torch.cfloat))
            self.bias1 = nn.Parameter(self.scale * torch.rand(self.c_out, self.modes1*2, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        x = compl_mul2d(x, self.weights1) + self.bias1
        x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        # x = torch.tanh(x.abs()) * (x / (x.abs() + self.eps))
        return x


class R_transformations(nn.Module):
    def __init__(self, width_list, modes1_list, modes2_list, depth, mlp):
        super(R_transformations, self).__init__()
        self.depth = depth
        self.modes1_list = modes1_list
        self.modes2_list = modes2_list
        self.width_list = width_list
        self.n_feature = 3
        self.features = 4*self.n_feature
        self.channelexp = nn.ModuleList([
            nn.ModuleList([R_trans(width_list[0]+self.features, width_list[0], modes1_list[0], modes2_list[0], mlp), ]),
        ])
        for i in range(1,self.depth):
            self.channelexp.append(nn.ModuleList([
                R_trans(width_list[i-1]+self.features, width_list[i], modes1_list[i], modes2_list[i], mlp),
                # R_trans(width_list[i], width_list[i], modes1_list[i], modes2_list[i], mlp),
                R_trans(width_list[i], width_list[i-1], modes1_list[i], modes2_list[i], mlp), ]),)

    def init_skip_in(self, x):
        skip = []
        for i in range(self.depth):
            skip.append(torch.zeros(x.shape[0], self.width_list[i], self.modes1_list[i]*2, self.modes2_list[i], dtype=torch.cfloat, device=x.device))
        return skip

    def down_block(self, i, x1, K_features, skip_in):
        m1, m2 = self.modes1_list[i], self.modes2_list[i]
        x2 = torch.cat((x1[:, :, :m1, :m2], x1[:, :, -m1:, :m2]), dim=2)  # (width, mode/2)
        K_features = torch.cat((K_features[:, :, :m1, :m2], K_features[:, :, -m1:, :m2]), dim=2) # (width, mode/2)
        x2 = torch.cat([x2, K_features], dim=1)
        x2 = self.channelexp[i][0](x2)  # (width*2, mode/2)
        x2 = (x2 + skip_in)
        # x2 = self.channelexp[i][1](x2) #(width*2, mode/2)
        return x2

    def up_add_corner(self, i, x2, x1):
        m1, m2 = self.modes1_list[i], self.modes2_list[i]
        x1[:, :, :m1, :m2] = (x1[:, :, :m1, :m2] + x2[:, :, :m1, :m2])
        x1[:, :, -m1:, :m2] = (x1[:, :, -m1:, :m2] + x2[:, :, -m1:, :m2])
        return x1

    def forward(self, x, K_features, skip_in=None):
        # adaptive modes
        M1, M2 = x.shape[2]//2, x.shape[3]
        self.modes1_list = []
        self.modes2_list = []
        for l in range(self.depth):
            n = 2 ** l
            self.modes1_list.append(M1 // n)
            self.modes2_list.append(M2 // n)

        if skip_in == None:
            skip_in = self.init_skip_in(x)

        if x.shape[2] != self.modes1_list[0]:
            x = torch.cat((x[:, :, :self.modes1_list[0], :self.modes2_list[0]], x[:, :, -self.modes1_list[0]:, :self.modes2_list[0]]), dim=2)  # (width, mode/2)
            K_features = torch.cat((K_features[:, :, :self.modes1_list[0], :self.modes2_list[0]], K_features[:, :, -self.modes1_list[0]:, :self.modes2_list[0]]), dim=2)

        # down
        x1 = x
        x1 = (x1 + skip_in[0])
        x1 = torch.cat([x1, K_features], dim=1)
        x1 = self.channelexp[0][0](x1)  # (width, mode)
        x = [x1,]
        skip_out = [x1,]

        for i in range(1,self.depth):
            x1 = self.down_block(i, x1, K_features, skip_in[i])
            x.append(x1)
            skip_out.append(x1)

        # up
        for i in range(self.depth-1, 0, -1):
            x[i] = self.channelexp[i][-1](x[i])  # (width*2, mode/4)
            x[i-1] = self.up_add_corner(i, x[i], x[i-1])  # (width*2, mode/2)

        x = x[0]  # (width, mode)
        return x, skip_out

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


class FNO_U(nn.Module):
    def __init__(self, modes1_list, modes2_list, width_list, depth, mlp, in_channel=9, out_channel=1):
        super(FNO_U, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_c = in_channel
        self.out_c = out_channel
        self.modes1_list = modes1_list
        self.modes2_list = modes2_list
        self.width_list = width_list
        self.width = width_list[0]
        self.depth = depth
        self.layer = 4
        self.pad = 8 # 1/1, 1/2, 1/4, 1/8
        # self.adain = AdaIN()

        self.p = MLP(self.in_c, self.width, self.width) # input channel is 3: (a(x, y), x, y)
        # self.feat_p = nn.Conv2d(1, self.width, kernel_size=1) # with k is 13

        self.k = K() # and then pass this into FFT_Down

        self.R0 = R_transformations(self.width_list, self.modes1_list, self.modes2_list, depth, mlp)
        self.R1 = R_transformations(self.width_list, self.modes1_list, self.modes2_list, depth, mlp)
        self.R2 = R_transformations(self.width_list, self.modes1_list, self.modes2_list, depth, mlp)
        self.R3 = R_transformations(self.width_list, self.modes1_list, self.modes2_list, depth, mlp)

        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)

        self.w = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1),
            nn.Conv2d(self.width, self.width, 1),
            nn.Conv2d(self.width, self.width, 1),
            nn.Conv2d(self.width, self.width, 1),
        ])

        self.s = nn.ModuleList([])
        for l in range(self.layer):
            self.s.append(nn.ModuleList([]))
            for i in range(self.depth):
                if l < self.layer - 1:
                    self.s[l].append(MLP(self.width_list[i], self.width_list[i], self.width_list[i]))
                elif l == self.layer - 1:
                    self.s[l].append(nn.Conv2d(self.width_list[i], self.width, 1, dtype=torch.cfloat))

        self.q = MLP(self.width*(1+self.depth), self.out_c, self.width * 4) # output channel is 1: u(x, y)

    def get_k(self, x, Re, S1, S2):
        k_mat = self.k.get_k(x.size(0), S1, S2).to(x.device)
        # scale k
        # Re = Re.view(k_mat.shape[0], 1, 1, 1).cuda()
        # k_mat = torch.div(k_mat, Re*.01)
        # x_ft = torch.concat((x_ft, k_mat[:, :, :, :x_ft.size(-1)]), 1) # just take real components of k_mat

        # append Re and k
        Re = Re.reshape(x.shape[0], 1, 1, 1)
        # Re_mat = torch.ones((x_ft.shape[0], 1, x_ft.shape[2], x_ft.shape[3])).to(x.device)
        # Re_mat = torch.mul(Re_mat, Re)
        Re_mat = self.k.get_Re(Re, S1, S2).to(x.device)
        # k_mat = torch.ones_like(k_mat)
        # Re_mat = torch.ones_like(Re_mat)
        features = torch.concat((k_mat, Re_mat), 1) # (B, 6, H, W)
        return features


    def fft(self, x):
        batchsize, C, S1, S2 = x.shape
        # x = torch.cat([x, -x.flip(dims=[-2])[..., 1:S1-1:self.pad, :]], dim=-2)
        # x = torch.cat([x, -x.flip(dims=[-1])[..., :, 1:S2-1:self.pad]], dim=-1)
        x_ft = torch.fft.rfft2(x, dim=[2,3], norm="ortho")
        return x_ft

    def ifft(self, x):
        x = torch.fft.irfftn(x, s=(self.S1_extended, self.S2_extended), norm="ortho")
        # x = x[:, :, :self.S1, :self.S2]
        return x

    def skip_block(self, skip, l):
        for i in range(self.depth):
            x = skip[i]
            x = torch.fft.irfft2(x, norm="ortho")
            x = self.s[l][i](x)
            x = torch.fft.rfft2(x, norm="ortho")
            skip[i] = x
        return skip

    def skip_end(self, out, skip):
        for i in range(self.depth):
            x = skip[i]
            x = self.s[-1][i](x)
            x = self.ifft(x)
            out = torch.cat([out, x], dim=1)
        return out

    def forward(self, x, Re):
        std = torch.std(x[:,1:].clone().detach(), dim=[1,2,3], keepdim=True)
        x = torch.cat([x[:, :1], x[:, 1:] / std], dim=1)

        S1, S2 = x.shape[2], x.shape[3]
        self.S1, self.S2 = S1, S2
        # self.S1_extended, self.S2_extended = S1+int(np.ceil((S1-2)/self.pad)), S2+int(np.ceil((S2-2)/self.pad))
        self.S1_extended, self.S2_extended = self.S1, self.S2

        grid = self.get_grid(x.shape, x.device)
        K_features = self.get_k(x, Re, self.S1_extended, self.S2_extended)
        x = torch.cat((x, grid), dim=1)
        x = self.p(x)

        # first fno block
        w = x
        x = self.fft(x)
        x, skip = self.R0(x, K_features)
        x = self.ifft(x)
        skip = self.skip_block(skip, 0)
        x = self.mlp0(x)
        w = self.w[0](w)
        x = x + w
        x = F.gelu(x)

        # second fno block
        w = x
        x = self.fft(x)
        x, skip = self.R1(x, K_features, skip)
        x = self.ifft(x)
        skip = self.skip_block(skip, 1)
        x = self.mlp1(x)
        w = self.w[1](w)
        x = x + w
        x = F.gelu(x)

        # third fno block
        w = x
        x = self.fft(x)
        x, skip = self.R2(x, K_features, skip)
        x = self.ifft(x)
        skip = self.skip_block(skip, 2)
        x = self.mlp2(x)
        w = self.w[2](w)
        x = x + w
        x = F.gelu(x)

        # fourth fno block
        w = x
        x = self.fft(x)
        x, skip = self.R3(x, K_features, skip)
        x = self.ifft(x)
        x = self.mlp3(x)
        w = self.w[3](w)
        x = x + w

        x = self.skip_end(x, skip)
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

