"""
@author: Zongyi Li and Catherine Deng

multiband FNO, with Re / k, mlp in channel expansion
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from symmetry_no.models.disco_conv import EquidistantDiscreteContinuousConv2d

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

# f domain, multiplies modes by a weight matrix
class R_trans(nn.Module):
    def __init__(self, c_in, c_out, modes1, modes2, mlp=False, act=True): # TODO: double check this
        super(R_trans, self).__init__()

        self.c_in = int(c_in)
        self.c_out = int(c_out)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.mlp = mlp
        self.eps = 0.0001
        self.scale = (1 / (c_in * c_out))
        self.act = act

        self.norm_real = nn.GroupNorm(1, c_out)
        self.norm_imag = nn.GroupNorm(1, c_out)

        if self.mlp:
            self.weights = nn.Parameter(self.scale * torch.rand(self.c_in, self.c_out, dtype=torch.cfloat))
            # self.bias = nn.Parameter(self.scale * torch.rand(self.c_out, 1, 1, dtype=torch.cfloat))
        else:
            self.weights = nn.Parameter(self.scale * torch.rand(self.c_in, self.c_out, self.modes1*2, self.modes2+1, dtype=torch.cfloat))
            # self.bias = nn.Parameter(self.scale * torch.rand(self.c_out, self.modes1*2, self.modes2+1, dtype=torch.cfloat))

    def forward(self, x):
        if self.mlp:
            x = compl_mul2d(x, self.weights)
        else:
            weights = self.weights
            # align input and weight, if input is smaller, we truncate the weight
            if x.shape[-2] < self.modes1*2:
                weights = torch.cat([weights[..., :x.shape[-2]//2,:], weights[..., -x.shape[-2]//2:,:]], dim=-2)
            if x.shape[-1] < self.modes2+1:
                weights = weights[..., :x.shape[-1]]
            x = compl_mul2d(x, weights)

        if self.act:
            # x_real = self.norm_real(x.real)
            # x_imag = self.norm_imag(x.imag)
            x = F.gelu(x.real) + 1j * F.gelu(x.imag)
            # x = torch.tanh(x.abs()) * (x / (x.abs() + self.eps))
        return x


class R_transformations(nn.Module):
    def __init__(self, width_list, modes1_list, modes2_list, level, mlp, n_feature=3):
        super(R_transformations, self).__init__()
        self.level = level
        self.modes1_list = modes1_list
        self.modes2_list = modes2_list
        self.width_list = width_list
        self.mlp = mlp
        self.n_feature = n_feature
        self.features = 3*self.n_feature
        self.sparsity_threshold = 0.05
        self.channelexp = nn.ModuleList([
            nn.ModuleList([R_trans(width_list[0], width_list[0], modes1_list[0], modes2_list[0], mlp=True), ]),
        ])
        self.p = MLP(self.features, width_list[0], width_list[0], dtype=torch.cfloat)  # k_mat + Re

        for i in range(1,self.level):
            self.channelexp.append(nn.ModuleList([
                R_trans(width_list[i-1], width_list[i], modes1_list[i], modes2_list[i], mlp),
                # R_trans(width_list[i], width_list[i], modes1_list[i], modes2_list[i], mlp),
                R_trans(width_list[i], width_list[i-1], modes1_list[i], modes2_list[i], mlp), ]),)

    def init_skip_in(self, x, modes1_list, modes2_list):
        skip = []
        skip.append(torch.ones(x.shape[0], self.width_list[0], x.shape[2], x.shape[3], dtype=torch.cfloat, device=x.device))
        for i in range(1, self.level):
            skip.append(torch.ones(x.shape[0], self.width_list[i], modes1_list[i]*2, modes2_list[i], dtype=torch.cfloat, device=x.device))
        return skip

    def down_block(self, i, x1, skip_in, m1, m2):
        """
        down block
        input x1 (B, C, H, W)
        output x2 (B, C, H//2, W//2)
        """
        x2 = torch.cat((x1[:, :, :m1, :m2], x1[:, :, -m1:, :m2]), dim=2)  # (width, mode/2)
        x2 = self.channelexp[i][0](x2)  # (width*2, mode/2)
        x2 = (x2 + skip_in)
        return x2

    def up_add_corner(self, x2, x1, m1, m2):
        """
        up block
        input: x1 (B, C, H, W), x2 (B, C, H//2, W//2)
        output: x1 (B, C, H, W)
        """
        x1[:, :, :m1, :m2] = (x1[:, :, :m1, :m2] + x2[:, :, :m1, :m2])
        x1[:, :, -m1:, :m2] = (x1[:, :, -m1:, :m2] + x2[:, :, -m1:, :m2])
        return x1

    def get_k(self, batchsize, S1, S2, device):
        # print("computing k")
        modes1, modes2 = S1 // 2, S2 // 2
        k_x1 = torch.cat((torch.arange(start=0, end=S1 - modes1, step=1, device=device),
                          torch.arange(start=-(modes1), end=0, step=1, device=device)), 0).\
                            reshape(S1, 1).repeat(1,S2).reshape(1, S1, S2)
        k_x2 = torch.cat((torch.arange(start=0, end=S2 - modes2, step=1, device=device),
                          torch.arange(start=-(modes2), end=0, step=1, device=device)), 0).\
                            reshape(1, S2).repeat(S1,1).reshape(1, S1, S2)

        k = torch.cat((k_x1, k_x2), 0)[..., :modes2 + 1]
        k = k.reshape(1, 2, S1, modes2 + 1)

        # feature embedding
        pow = torch.arange(start=1, end=self.n_feature+1, device=device).reshape(self.n_feature, 1, 1, 1) /self.n_feature
        k_mat = torch.pow(k.cfloat(), pow)
        k_mat = k_mat.reshape(1, 2*self.n_feature, S1, modes2 + 1).repeat(batchsize, 1,1,1)
        k_mat.requires_grad = False
        return k_mat

    def embed_re(self, re, S1, S2):
        pow = torch.arange(start=1, end=self.n_feature+1, device=re.device).reshape(1, self.n_feature, 1, 1) /self.n_feature
        re_mat = torch.pow(re, pow).repeat(1,1, S1, S2 // 2 +1)
        return re_mat

    def forward(self, x, Re, S1, S2, skip_in=None):
        # adaptive modes
        batchsize = x.shape[0]
        M1, M2 = S1//2, S2//2 + 1
        modes1_list = self.modes1_list
        modes2_list = self.modes2_list

        # first level is mlp
        modes1_list[0] = M1
        modes2_list[0] = M2
        # in tensor implementation, if input is larger than weight, set the mode_list with the weight
        if not self.mlp:
            for l in range(0, self.level):
                if M1 < self.modes1_list[l]:
                    modes1_list[l] = M1
                if M2 < self.modes1_list[l]:
                    modes2_list[l] = M2

        # if the x is larger than weight, truncate x
        # if x.shape[-2] > modes1_list[0]*2 or x.shape[-1] > modes2_list[0]+1:
        #     x = torch.cat((x[:, :, :modes1_list[0], :modes2_list[0]], x[:, :, -modes1_list[0]:, :modes2_list[0]]), dim=-2)  # (width, mode/2)
        #     K_features = torch.cat((K_features[:, :, :modes1_list[0], :modes2_list[0]], K_features[:, :, -modes1_list[0]:, :modes2_list[0]]), dim=-2)

        if skip_in == None:
            skip_in = self.init_skip_in(x, modes1_list, modes2_list)

        # add wavenumber k_mat
        k_mat = self.get_k(batchsize, S1, S2, device=x.device)
        re_mat = self.embed_re(Re, S1, S2)
        re_feature = self.p(torch.cat([k_mat, re_mat], dim=1))
        x = x * re_feature
        x = x * (x.abs() > self.sparsity_threshold)

        # down
        x1 = (x + skip_in[0])
        x1 = self.channelexp[0][0](x1)  # (width, mode)

        x = [x1,]
        skip_out = [x1,]

        for i in range(1,self.level):
            x1 = self.down_block(i, x1, skip_in[i], modes1_list[i], modes2_list[i])
            x.append(x1)
            skip_out.append(x1)

        # up
        for i in range(self.level-1, 0, -1):
            x[i] = self.channelexp[i][-1](x[i])  # (width*2, mode/4)
            x[i-1] = self.up_add_corner(x[i], x[i-1], modes1_list[i], modes2_list[i])  # (width*2, mode/2)

        x = x[0]  # (width, mode)
        return x, skip_out

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dtype=torch.float):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1, padding="same", dtype=dtype)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1, padding="same", dtype=dtype)

    def forward(self, x):
        x = self.mlp1(x)
        if torch.is_complex(x):
            x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        else:
            x = F.gelu(x)
        x = self.mlp2(x)
        return x

class Local(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, in_shape=64, dtype=torch.float):
        super(Local, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding="same", dtype=dtype)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding="same", dtype=dtype)
        self.size = in_shape

    def forward(self, x):
        input_size = [x.shape[-2], x.shape[-1]]
        if input_size[1] != self.size:
            x = F.interpolate(x, size=self.size)

        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)

        if input_size != self.size:
            x = F.interpolate(x, size=input_size)
        return x

class Local_disco(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, in_shape=64, dtype=torch.float):
        super(Local, self).__init__()
        self.mlp1 = EquidistantDiscreteContinuousConv2d(in_channels, mid_channels, kernel_shape=3, in_shape=[in_shape,in_shape])
        self.mlp2 = EquidistantDiscreteContinuousConv2d(mid_channels, out_channels, kernel_shape=3, in_shape=[in_shape,in_shape])

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO_U(nn.Module):
    def __init__(self, modes1_list, modes2_list, width_list, level, mlp, depth=3, in_channel=9, out_channel=1, n_feature=5, boundary=False):
        super(FNO_U, self).__init__()

        """
        The overall network. It contains 4 depths of the Fourier depth.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 depths of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_c = in_channel + 2
        self.out_c = out_channel
        self.modes1_list = modes1_list
        self.modes2_list = modes2_list
        self.width_list = width_list
        self.width = width_list[0]
        self.level = level
        self.depth = depth
        self.pad = 8 # 1/1, 1/2, 1/4, 1/8
        self.n_feature = n_feature
        self.size = 128
        self.boundary = boundary

        self.p = MLP(self.in_c, self.width, 2*self.width) # input channel is 3: (a(x, y), x, y)

        self.R = nn.ModuleList([])
        self.mlp = nn.ModuleList([])
        self.w = nn.ModuleList([])
        self.norm1 = nn.ModuleList([])
        self.norm2 = nn.ModuleList([])
        for l in range(self.depth):
            self.R.append(R_transformations(self.width_list, self.modes1_list, self.modes2_list, level, mlp, self.n_feature))
            self.mlp.append(MLP(self.width, self.width, self.width))
            # self.mlp.append(Local(self.width, self.width, self.width, in_shape=self.modes1_list[0]*2))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
            self.norm1.append(nn.GroupNorm(1, self.width))
            self.norm2.append(nn.GroupNorm(1, self.width))

        self.q = MLP(self.width, self.out_c, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x, re):

        if self.boundary:
            std = torch.std(x[:,1:].clone(), dim=[1,2,3], keepdim=True)
            x = torch.cat([x[:, :1], x[:, 1:] / std], dim=1)
        if re==None:
            re = torch.ones(x.shape[0], device=x.device)

        S1, S2 = x.shape[2], x.shape[3]
        # self.S1_extended, self.S2_extended = S1+int(np.ceil((S1-2)/self.pad)), S2+int(np.ceil((S2-2)/self.pad))

        re = re.reshape(-1, 1, 1, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1) # 1 is the channel dimension
        x = self.p(x)

        # first fno block
        skip = None

        for i in range(self.depth):
            x1 = self.w[i](x)
            x = self.norm1[i](x)
            x_ft = torch.fft.rfft2(x, dim=[2,3], norm="forward")
            x_ft, skip = self.R[i](x_ft, re, S1, S2, skip)
            x = torch.fft.irfftn(x_ft, s=(S1, S2), norm="forward")
            x = self.norm2[i](x)
            x2 = self.mlp[i](x)
            x = x1 + x2
            if i != self.depth-1:
                x = F.gelu(x)

        x = self.q(x)

        if self.boundary:
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
