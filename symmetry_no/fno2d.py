"""

Taken from 
[https://github.com/neuraloperator/neuraloperator/blob/master/fourier_2d.py]
@author: Zongyi Li (with some minor tweaks by Sam Lanthaler)

as discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).

"""

import sys
import torch.nn.functional as F
from timeit import default_timer
from symmetry_no.utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

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

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, depth=4, in_channel=7, out_channel=1):
        super(FNO2d, self).__init__()

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

        self.p = MLP(self.in_channel, self.width, self.width)
        #self.p = nn.Linear(self.in_channel, self.width) # input channel is 7: (a(x, y), BC_left, BC_bottom, BC_right, BC_top, x, y)

        self.conv = []
        self.mlp = []
        self.w = []
        for _ in range(depth):
            self.conv.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.mlp.append(MLP(self.width, self.width, self.width))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
        #
        self.conv = nn.ModuleList(self.conv)
        self.mlp = nn.ModuleList(self.mlp)
        self.w = nn.ModuleList(self.w)
        #
        self.q = MLP(self.width, self.out_channel, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        # normalize the input
        std = torch.std(x[:,1:,:,:].clone().detach(), dim=[1,2,3], keepdim=True)
        std.requires_grad = False
        x[:, 1:,:,:] = x[:,1:,:,:] / std  ## this causes memory leak??
        #print(torch.cuda.memory_summary())
        
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

#    def forward(self, x):
#        grid = self.get_grid(x.shape, x.device)
#        x = torch.cat((x, grid), dim=1) # 1 is the channel dimension
#        x = self.p(x.permute(0,2,3,1)).permute(0,3,1,2)
#        #x = x.permute(0, 3, 1, 2)
#
#        for i in range(self.depth):
#            x1 = self.conv[i](x)
#            x1 = self.mlp[i](x1)
#            x2 = self.w[i](x)
#            x = x1 + x2
#            x = F.gelu(x)
#
#        x = self.q(x)
#        #x = x.permute(0, 2, 3, 1)
#        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)











    

class FNO2dBasic(nn.Module):
    def __init__(self, modes1, modes2,  width, depth=4, in_channel=1, out_channel=1):
        super(FNO2dBasic, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function, boundary conditions and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=1)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.p = MLP(self.in_channel+2, self.width, self.width)
        #self.p = nn.Linear(self.in_channel, self.width) # input channel is 7: (a(x, y), BC_left, BC_bottom, BC_right, BC_top, x, y)

        self.conv = []
        self.mlp = []
        self.w = []
        for _ in range(depth):
            self.conv.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.mlp.append(MLP(self.width, self.width, self.width))
            self.w.append(nn.Conv2d(self.width, self.width, 1))
        #
        self.conv = nn.ModuleList(self.conv)
        self.mlp = nn.ModuleList(self.mlp)
        self.w = nn.ModuleList(self.w)
        #
        self.q = MLP(self.width, self.out_channel, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
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
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    

    
class FNO2dSemiLinear(nn.Module):
    '''
    Implements a FNO2d to approximate operators Phi of a semi-linear form
    
    Phi(a,c1*g1+c2*g2) = c1 * Phi(a,g1) + c2 * Phi(a,g2),

    where g is a one-dimensional input defined over an interval, e.g. representing the boundary condition.

    The structure is as follows. 

    1. We project g onto a fixed basis phi_k,
    
    g(x) = \sum_{k=1}^K g_k phi_k(x),

    e.g. phi_k = Fourier basis --> g_k = Fourier coefficients

    2. We have a standard FNO2d with K output channels,

    FNO(a) = [FNO(a)_1, ..., FNO(a)_K]

    3. We combine them in a linear way

    output = \sum_{k=1}^K g_k * FNO(a)_k

    -------------------------------
    Example: Darcy flow with BC, and domain = the square [0,1]^2
    
    -div(a * grad(u)) = 0
    u|_{boundary} = g

    This is non-linear in a, but linear in g.
    '''

    def __init__(self, modes1, modes2,  width, depth=4,
                 in_channel=1, out_channel=1, lin_modes=12):
        super(FNO2dSemiLinear, self).__init__()

        """
        input: coefficient function, boundary conditions (a, BC1)
        input shape: (batchsize, x=s, y=s, c=1), (batchsize, x=s, c=1)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lin_modes = lin_modes

        # non-linear part of the architecture
        # output shape: (batchsize, x=s, y=s, c=fno_out_channel)
        if out_channel!=1:
            raise NotImplementedError(f'{out_channel=} not yet supported. Only out_channel==1 is allowed right now!')
        # explanation:
        # 1. factor 2 because Fourier coefficients complex --> real transformation
        # 2. factor 4 because we concatenate 4 boundary conditions into a (4, x=s) tensor
        self.lin_channel = out_channel * lin_modes * 2 * 4 # explanation above
        self.fno_out_channel = self.lin_channel #16
        self.fno = FNO2dBasic(modes1, modes2, width, depth, in_channel, self.fno_out_channel)

#        self.fno = FNO2dBasic(modes1, modes2, width, depth, 5, 1) 

        # 
        #self.proj = nn.Parameter(torch.randn(self.lin_channel, self.fno_out_channel)) 
        
        
    def forward(self, x):
        ''' 
        To be consistent with other models, we keep the structure
of the inputs. This means that x has shape (batchsize, channels, x=s, y=s).

        '''
        #
        batchsize, gridsize = x.shape[0], x.shape[-1]
        nonlinear = self.fno(x[:,0,:,:].unsqueeze(1))
        
        # extract BC
        x_ = x.clone()
        BC_left   = x_[:,1, 0, :].unsqueeze(1) # shape: (B, 1, s)
        BC_bottom = x_[:,2, :, 0].unsqueeze(1)
        BC_right  = x_[:,3,-1, :].unsqueeze(1)
        BC_top    = x_[:,4, :,-1].unsqueeze(1)
        # combine all BC's
        BC = torch.cat((BC_left, BC_bottom, BC_right, BC_top), dim=1) # shape: (B, 4, s)
        #
        BC_ft = torch.fft.rfft(BC, dim=-1, norm='forward') # shape: (B, 4, s//2+1) -- set norm to approximately compute g_k = \int_0^1 g(x)e^{-2\pi kx} dx
        BC_ft = BC_ft[:,:,:self.lin_modes] # retain only specified number of modes
        BC_ft = torch.view_as_real(BC_ft)  # get real-valued array
        # combine 4 BC channels, with 2*lin_modes 
        BC_ft = BC_ft.reshape(batchsize, -1)  # shape: (B, 8*lin_modes)
        
        # combine into semi-linear output
        output = torch.einsum('bjxy,bj->bxy',nonlinear,BC_ft) / (8*self.lin_modes)
        output = output.unsqueeze(1) # add channel dimension
        return output
       
