import torch
import math

from timeit import default_timer


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, exp=False, DCT=True, device=None):

        self.dim = dim
        self.device = device
        self.DCT = DCT

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        if DCT:
            size = size *2

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

            if exp:
                self.sqrt_eig = (size ** 2) * torch.exp( -(sigma * torch.sqrt(k_x**2 + k_y**2)) ** alpha)
            else:
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
        random_field = torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1)))

        if self.DCT:
            random_field = random_field[..., :random_field.shape[-2]//2, :random_field.shape[-1]//2]

        return random_field




from symmetry_no.helmholtz_utilities import HelmholtzExtractBC
from symmetry_no.darcy_utilities import DarcyExtractBC


def sample_Darcy(rate, input=None, alpha_a=0.5, alpha_g=1, sigma_g=1, keepsize=False):
    device = input.device
    N = input.shape[0]
    size_input = input.shape[2]
    if keepsize:
        size = size_input
    else:
        size = math.floor(size_input * rate) //2 *2
    rate = size/size_input

    sigma = 1/rate
    a_GRF = GaussianRF(dim=2, size=size, alpha=alpha_a, sigma=sigma, exp=True, device=device)
    a = a_GRF.sample(N).real.to(device)
    a[a > 0] = 12
    a[a <= 0] = 2
    g_GRF = GaussianRF(dim=2, size=size, alpha=alpha_g, sigma=sigma_g, exp=True, device=device)
    g = g_GRF.sample(N).real.to(device)
    g = g / torch.std(g) * torch.std(input[:, 1:])

    x = torch.zeros(N, 5, size, size, dtype=torch.float32, device=device)
    x[:, 0] = a
    new_input = DarcyExtractBC(x, g).to(device)

    return new_input, rate

def sample_helm(rate, input, alpha_a=1.5, alpha_g=3.5, tau=2, keepsize=False):
    device = input.device
    N = input.shape[0]
    size_input = input.shape[2]
    size = math.floor(size_input * rate) //2 *2
    rate = size/size_input
    repeat = math.ceil(size / size_input)

    a_GRF = GaussianRF(dim=2, size=size, alpha=alpha_a, tau=tau, device=device)
    a = a_GRF.sample(N).real
    a = 1 + 0.5*11*(1 + torch.tanh(a*100))
    g_GRF = GaussianRF(dim=2, size=size, alpha=alpha_g, tau=tau, device=device)
    g = g_GRF.sample(N).to(device)
    g = torch.view_as_real(g).permute(0, 3, 1, 2)
    g = g / torch.std(g) * torch.std(input[:, 1:])


    # For Helmholtz we repeat the input for larger size
    input = input.repeat(1, 1, repeat, repeat)[:, :, :size, :size]

    x = torch.zeros(N, 9, size, size, dtype=torch.float32, device=device)
    x[:, 0] = a
    new_input = HelmholtzExtractBC(x, g)
    new_input[:, 1:] = input[:, 1:] + new_input[:, 1:]

    if keepsize:
        # new_input = torch.nn.functional.interpolate(new_input,
        #                                 size=(size_input, size_input),
        #                                 mode='bilinear',
        #                                 align_corners=True)
        sub = max(repeat-1, 1)
        new_input = new_input[..., ::sub, ::sub][:, :, :size_input, :size_input]

    return new_input.to(device), rate


def sample_NS(rate, input, alpha_a=2, alpha_g=2, tau=2, sample_type="interp", keepsize=False, maxsize=256):
    device = input.device
    N = input.shape[0]
    C = input.shape[1]
    size_input = input.shape[2]
    rate = rate.to(device)
    if keepsize:
        size = size_input
    elif maxsize is not None:
        size = maxsize
    else:
        size = math.floor(size_input * rate) // 2 *2
        # rate = size/size_input # adject f

    a_GRF = GaussianRF(dim=2, size=size, alpha=0.5, sigma=4/rate, exp=True, device=device)
    # a_GRF = GaussianRF(dim=2, size=size, alpha=alpha_a, tau=tau, device=device)
    a = a_GRF.sample(N*C).real.to(device).reshape(N, C, size, size)
    a = 0.1 * a / torch.std(a) * torch.std(input)

    # g_GRF = GaussianRF(dim=2, size=size, alpha=alpha_g, tau=tau)
    # g = g_GRF.sample(N).real.to(device)
    # g = g / torch.std(g) * torch.std(input[:, 1:])

    ### repeat
    #repeat = math.ceil(size / size_input)
    #input = input.repeat(1, 1, repeat, repeat)[:, :, :size, :size]

    ### flip
    if sample_type == "flip":
        while input.shape[-1] < size:
            input = torch.cat([input, input.flip(-2)], dim=-2)
            input = torch.cat([input, input.flip(-1)], dim=-1)
        input = input[..., :size, :size]
    ### interp
    elif sample_type == "interp":
        if input.shape[-1] != size:
            input = torch.nn.functional.interpolate(input,
                                            size=(size, size),
                                            mode='bilinear',
                                            align_corners=True)

    new_input = input + a
    return new_input, rate
