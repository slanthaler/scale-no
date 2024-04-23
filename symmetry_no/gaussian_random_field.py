import torch
import math

from timeit import default_timer


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1)))


from symmetry_no.helmholtz_utilities import HelmholtzExtractBC
from symmetry_no.darcy_utilities import DarcyExtractBC


def sample_Darcy(rate, input=None, alpha_a=1.5, alpha_g=3.5, tau=2):
    device = input.device
    N = input.shape[0]
    size_input = input.shape[2]
    size = math.floor(size_input * rate) //2 *2
    rate = size/size_input

    a_GRF = GaussianRF(dim=2, size=size, alpha=alpha_a, tau=tau)
    a = a_GRF.sample(N).real
    g_GRF = GaussianRF(dim=2, size=size, alpha=alpha_g, tau=tau)
    g = g_GRF.sample(N).real

    x = torch.zeros(N, 5, size, size, dtype=torch.float32, device=device)
    x[:, 0] = a
    new_input = DarcyExtractBC(x, g)
    return new_input.to(device), rate

def sample_helm(rate, input, alpha_a=1.5, alpha_g=3.5, tau=2):
    device = input.device
    N = input.shape[0]
    size_input = input.shape[2]
    size = math.floor(size_input * rate) //2 *2
    rate = size/size_input
    repeat = math.ceil(size / size_input)

    a_GRF = GaussianRF(dim=2, size=size, alpha=alpha_a, tau=tau)
    a = a_GRF.sample(N).real
    a = 1 + 0.5*11*(1 + torch.tanh(a*100))
    g_GRF = GaussianRF(dim=2, size=size, alpha=alpha_g, tau=tau)
    g = g_GRF.sample(N) / torch.std(g)
    g = torch.view_as_real(g).permute(0, 3, 1, 2)

    # For Helmholtz we repeat the input for larger size
    input = input.repeat(1, 1, repeat, repeat)[:, :, :size, :size]

    x = torch.zeros(N, 9, size, size, dtype=torch.float32, device=device)
    x[:, 0] = a
    new_input = HelmholtzExtractBC(x, g)
    new_input[:, 1:] = input[:, 1:] * new_input[:, 1:]
    return new_input.to(device), rate


def sample_NS(rate, input, alpha_a=1.5, alpha_g=3.5, tau=2):
    device = input.device
    N = input.shape[0]
    size_input = input.shape[2]
    size = math.floor(size_input * rate) //2 *2
    rate = size/size_input
    repeat = math.ceil(size / size_input)

    a_GRF = GaussianRF(dim=2, size=size, alpha=alpha_a, tau=tau)
    a = a_GRF.sample(N).real.to(device)
    a = a / torch.std(a)
    g_GRF = GaussianRF(dim=2, size=size, alpha=alpha_g, tau=tau)
    g = g_GRF.sample(N).real.to(device)
    g = g / torch.std(g)

    # For NS we interpolate the input for larger size
    # input = torch.nn.functional.interpolate(input,
    #                                 size=(size, size),
    #                                 mode='bilinear',
    #                                 align_corners=True)
    input = input.repeat(1, 1, repeat, repeat)[:, :, :size, :size]

    x = torch.zeros(N, 5, size, size, dtype=torch.float32, device=device)
    x[:, 0] = input[:, 0] + a
    new_input = DarcyExtractBC(x, g)
    new_input[:, 1:] = input[:, 1:] + new_input[:, 1:]
    return new_input.to(device), rate
