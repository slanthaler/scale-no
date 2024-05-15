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

# computes wavenumber matrix
class K(nn.Module):
    def __init__(self, n_feature=3):
        super(K, self).__init__()
        self.n_feature = n_feature


    def compute_k(self, S1, S2, Re=1):
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
        k_mat = k_mat * Re
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
            self.bias = nn.Parameter(self.scale * torch.rand(self.c_out, 1, 1, dtype=torch.cfloat))
        else:
            self.weights = nn.Parameter(self.scale * torch.rand(self.c_in, self.c_out, self.modes1*2, self.modes2+1, dtype=torch.cfloat))
            self.bias = nn.Parameter(self.scale * torch.rand(self.c_out, self.modes1*2, self.modes2+1, dtype=torch.cfloat))

    def forward(self, x):
        if self.mlp:
            x = compl_mul2d(x, self.weights) + self.bias
        else:
            weights = self.weights
            bias = self.bias
            # align input and weight, if input is smaller, we truncate the weight
            if x.shape[-2] < self.modes1*2:
                weights = torch.cat([weights[..., :x.shape[-2]//2,:], weights[..., -x.shape[-2]//2:,:]], dim=-2)
                bias = torch.cat([bias[..., :x.shape[-2]//2,:], bias[..., -x.shape[-2]//2:,:]], dim=-2)
            if x.shape[-1] < self.modes2+1:
                weights = weights[..., :x.shape[-1]]
                bias = bias[..., :x.shape[-1]]
            x = compl_mul2d(x, weights) + bias

        x_real = self.norm_real(x.real)
        x_imag = self.norm_imag(x.imag)

        if self.act:
            x = F.gelu(x_real) + 1j * F.gelu(x_imag)
            # x = torch.tanh(x.abs()) * (x / (x.abs() + self.eps))
        return x


class R_transformations(nn.Module):
    def __init__(self, width_list, modes1_list, modes2_list, depth, mlp, n_feature=3):
        super(R_transformations, self).__init__()
        self.depth = depth
        self.modes1_list = modes1_list
        self.modes2_list = modes2_list
        self.width_list = width_list
        self.mlp = mlp
        self.n_feature = n_feature
        self.features = 4*self.n_feature
        self.channelexp = nn.ModuleList([
            nn.ModuleList([R_trans(width_list[0]+self.features, width_list[0], modes1_list[0], modes2_list[0], mlp=True), ]),
        ])

        for i in range(1,self.depth):
            self.channelexp.append(nn.ModuleList([
                R_trans(width_list[i-1]+self.features, width_list[i], modes1_list[i], modes2_list[i], mlp),
                # R_trans(width_list[i], width_list[i], modes1_list[i], modes2_list[i], mlp),
                R_trans(width_list[i], width_list[i-1], modes1_list[i], modes2_list[i], mlp), ]),)

    def init_skip_in(self, x, modes1_list, modes2_list):
        skip = []
        skip.append(torch.ones(x.shape[0], self.width_list[0], x.shape[2], x.shape[3], dtype=torch.cfloat, device=x.device))
        for i in range(1, self.depth):
            skip.append(torch.ones(x.shape[0], self.width_list[i], modes1_list[i]*2, modes2_list[i], dtype=torch.cfloat, device=x.device))
        return skip

    def down_block(self, i, x1, K_features, skip_in, m1, m2):
        """
        down block
        input x1 (B, C, H, W)
        output x2 (B, C, H//2, W//2)
        """
        x2 = torch.cat((x1[:, :, :m1, :m2], x1[:, :, -m1:, :m2]), dim=2)  # (width, mode/2)
        K_features = torch.cat((K_features[:, :, :m1, :m2], K_features[:, :, -m1:, :m2]), dim=2) # (width, mode/2)
        x2 = torch.cat([x2, K_features], dim=1)
        x2 = self.channelexp[i][0](x2)  # (width*2, mode/2)
        x2 = (x2 + skip_in)
        # x2 = self.channelexp[i][1](x2) #(width*2, mode/2)
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

    def forward(self, x, K_features, skip_in=None):
        # adaptive modes
        M1, M2 = x.shape[2]//2, x.shape[3]
        modes1_list = self.modes1_list
        modes2_list = self.modes2_list

        # first level is mlp
        modes1_list[0] = M1
        modes2_list[0] = M2
        # in tensor implementation, if input is larger than weight, set the mode_list with the weight
        if not self.mlp:
            for l in range(0, self.depth):
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

        # down
        x1 = x
        x1 = (x1 + skip_in[0])
        x1 = torch.cat([x1, K_features], dim=1)
        # x1 = x1 * K_features
        x1 = self.channelexp[0][0](x1)  # (width, mode)
        x = [x1,]
        skip_out = [x1,]

        for i in range(1,self.depth):
            x1 = self.down_block(i, x1, K_features, skip_in[i], modes1_list[i], modes2_list[i])
            x.append(x1)
            skip_out.append(x1)

        # up
        for i in range(self.depth-1, 0, -1):
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
        # self.mlp1 = EquidistantDiscreteContinuousConv2d(in_channels, mid_channels, kernel_shape=3, in_shape=[in_shape,in_shape])
        # self.mlp2 = EquidistantDiscreteContinuousConv2d(mid_channels, out_channels, kernel_shape=3, in_shape=[in_shape,in_shape])
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding="same", dtype=dtype)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding="same", dtype=dtype)
        self.size = in_shape
        # self.norm = nn.GroupNorm(1, mid_channels)

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
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, in_shape=64, dtype=torch.float):
        super(Local, self).__init__()
        self.mlp1 = EquidistantDiscreteContinuousConv2d(in_channels, mid_channels, kernel_shape=3, in_shape=[in_shape,in_shape])
        self.mlp2 = EquidistantDiscreteContinuousConv2d(mid_channels, out_channels, kernel_shape=3, in_shape=[in_shape,in_shape])

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO_U(nn.Module):
    def __init__(self, modes1_list, modes2_list, width_list, depth, mlp, layer=3, in_channel=9, out_channel=1, n_feature=5):
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
        self.layer = layer
        self.pad = 8 # 1/1, 1/2, 1/4, 1/8
        self.n_feature = n_feature
        self.size = 128
        self.norm = nn.GroupNorm(1, self.width)

        self.p = MLP(self.in_c, self.width, 2*self.width) # input channel is 3: (a(x, y), x, y)
        self.k_mlp = MLP(4*self.n_feature, 4*self.n_feature, 2*self.width, dtype=torch.cfloat)


        self.k = K(self.n_feature) # and then pass this into FFT_Down
        self.R = nn.ModuleList([])
        self.mlp = nn.ModuleList([])
        self.w = nn.ModuleList([])
        self.s = nn.ModuleList([])

        for l in range(self.layer):
            self.R.append(R_transformations(self.width_list, self.modes1_list, self.modes2_list, depth, mlp, self.n_feature))
            # self.mlp.append(MLP(self.width, self.width, self.width))
            self.mlp.append(Local(self.width, self.width, self.width, in_shape=self.modes1_list[0]*2))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
            self.s.append(nn.ModuleList([]))
            for i in range(self.depth):
                if l < self.layer - 1:
                    # self.s[l].append(MLP(self.width_list[i], self.width_list[i], self.width_list[i]))
                    self.s[l].append(Local(self.width_list[i], self.width_list[i], self.width_list[i], in_shape=self.modes1_list[i]*2))
                elif l == self.layer - 1:
                    self.s[l].append(MLP(self.width_list[i], self.width, self.width_list[i]//2, dtype=torch.cfloat))
                    # self.s[l].append(Local(self.width_list[i], self.width, self.width_list[i]//2, in_shape=self.modes1_list[i]*2))

        self.q = MLP(self.width*(1+self.depth), self.out_c, self.width * 4) # output channel is 1: u(x, y)
        # self.q = Local(self.width*(1+self.depth), self.out_c, self.width * 4) # output channel is 1: u(x, y)

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
        features = self.k_mlp(features)
        return features

    def fft(self, x):
        # batchsize, C, S1, S2 = x.shape
        # x = torch.cat([x, -x.flip(dims=[-2])[..., 1:S1-1:self.pad, :]], dim=-2)
        # x = torch.cat([x, -x.flip(dims=[-1])[..., :, 1:S2-1:self.pad]], dim=-1)
        x_ft = torch.fft.rfft2(x, dim=[2,3], norm="backward")
        return x_ft

    def ifft(self, x, S1, S2):
        x = torch.fft.irfftn(x, s=(S1, S2), norm="backward")
        # x = x[:, :, :self.S1, :self.S2]
        return x

    def skip_block(self, skip, l):
        for i in range(self.depth):
            x = skip[i]
            x = torch.fft.irfft2(x, norm="backward")
            x = self.s[l][i](x)
            x = torch.fft.rfft2(x, norm="backward")
            skip[i] = x
        return skip

    def skip_end(self, out, skip, S1, S2):
        for i in range(self.depth):
            x = skip[i]
            x = self.s[-1][i](x)
            x = self.ifft(x, S1, S2)
            # x = self.s[-1][i](x)
            out = torch.cat([out, x], dim=1)
        return out

    def forward(self, x, re):

        x = x[:,0:1].repeat(1,5,1,1)

        if re==None:
            re = torch.ones(x.shape[0], device=x.device)
        if torch.mean(re) > 100:
            re = re / 100
        # re = re.reshape(-1, 1, 1, 1)

        S1, S2 = x.shape[2], x.shape[3]
        # self.S1_extended, self.S2_extended = S1+int(np.ceil((S1-2)/self.pad)), S2+int(np.ceil((S2-2)/self.pad))

        grid = self.get_grid(x.shape, x.device)
        K_features = self.get_k(x, re, S1, S2)
        # re_cat = re * torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), requires_grad=False, device=x.device)
        x = torch.cat((x, grid), dim=1) # 1 is the channel dimension
        x = self.p(x)

        # first fno block
        skip = None

        for i in range(self.layer):
            x1 = self.w[i](x)
            x2 = self.mlp[i](x)
            x = self.fft(x)
            x, skip = self.R[i](x, K_features, skip)
            x = self.ifft(x, S1, S2)
            x = x + x1 + x2
            x = self.norm(x)
            if i != self.layer-1:
                skip = self.skip_block(skip, i)
                x = F.gelu(x)

        x = self.skip_end(x, skip, S1, S2)
        x = self.q(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

