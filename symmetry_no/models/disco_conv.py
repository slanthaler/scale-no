import abc
from typing import List, Tuple, Union, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

def _compute_support_vals_isotropic(r: torch.Tensor, phi: torch.Tensor, nr: int, r_cutoff: float, norm: str = "s2"):
    """
    Computes the index set that falls into the isotropic kernel's support and returns both indices and values.
    """

    # compute the support
    dr = (r_cutoff - 0.0) / nr
    ikernel = torch.arange(nr).reshape(-1, 1, 1)
    ir = ikernel * dr

    if norm == "none":
        norm_factor = 1.0
    elif norm == "2d":
        norm_factor = math.pi * (r_cutoff * nr / (nr + 1))**2 + math.pi * r_cutoff**2 * (2 * nr / (nr + 1) + 1) / (nr + 1) / 3
    elif norm == "s2":
        norm_factor = 2 * math.pi * (1 - math.cos(r_cutoff - dr) + math.cos(r_cutoff - dr) + (math.sin(r_cutoff - dr) - math.sin(r_cutoff)) / dr)
    else:
        raise ValueError(f"Unknown normalization mode {norm}.")

    # find the indices where the rotated position falls into the support of the kernel
    iidx = torch.argwhere(((r - ir).abs() <= dr) & (r <= r_cutoff))
    vals = (1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr) / norm_factor
    return iidx, vals


def _compute_support_vals_anisotropic(r: torch.Tensor, phi: torch.Tensor, nr: int, nphi: int, r_cutoff: float, norm: str = "s2"):
    """
    Computes the index set that falls into the anisotropic kernel's support and returns both indices and values.
    """

    # compute the support
    dr = (r_cutoff - 0.0) / nr
    dphi = 2.0 * math.pi / nphi
    kernel_size = (nr - 1) * nphi + 1
    ikernel = torch.arange(kernel_size).reshape(-1, 1, 1)
    ir = ((ikernel - 1) // nphi + 1) * dr
    iphi = ((ikernel - 1) % nphi) * dphi

    if norm == "none":
        norm_factor = 1.0
    elif norm == "2d":
        norm_factor = math.pi * (r_cutoff * nr / (nr + 1))**2 + math.pi * r_cutoff**2 * (2 * nr / (nr + 1) + 1) / (nr + 1) / 3
    elif norm == "s2":
        norm_factor = 2 * math.pi * (1 - math.cos(r_cutoff - dr) + math.cos(r_cutoff - dr) + (math.sin(r_cutoff - dr) - math.sin(r_cutoff)) / dr)
    else:
        raise ValueError(f"Unknown normalization mode {norm}.")

    # find the indices where the rotated position falls into the support of the kernel
    cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
    cond_phi = (ikernel == 0) | ((phi - iphi).abs() <= dphi) | ((2 * math.pi - (phi - iphi).abs()) <= dphi)
    iidx = torch.argwhere(cond_r & cond_phi)
    vals = (1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr) / norm_factor
    vals *= torch.where(
        iidx[:, 0] > 0,
        (1 - torch.minimum((phi[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs(), (2 * math.pi - (phi[iidx[:, 1], iidx[:, 2]] - iphi[iidx[:, 0], 0, 0]).abs())) / dphi),
        1.0,
    )
    return iidx, vals

def _precompute_convolution_tensor_2d(grid_in, grid_out, kernel_shape, radius_cutoff=0.01, periodic=False):
    """
    Precomputes the translated filters at positions $T^{-1}_j \omega_i = T^{-1}_j T_i \nu$. Similar to the S2 routine,
    only that it assumes a non-periodic subset of the euclidean plane
    """

    # check that input arrays are valid point clouds in 2D
    assert len(grid_in) == 2
    assert len(grid_out) == 2
    assert grid_in.shape[0] == 2
    assert grid_out.shape[0] == 2

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-1]

    if len(kernel_shape) == 1:
        kernel_handle = partial(_compute_support_vals_isotropic, nr=kernel_shape[0], r_cutoff=radius_cutoff, norm="2d")
    elif len(kernel_shape) == 2:
        kernel_handle = partial(_compute_support_vals_anisotropic, nr=kernel_shape[0], nphi=kernel_shape[1], r_cutoff=radius_cutoff, norm="2d")
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    grid_in = grid_in.reshape(2, 1, n_in)
    grid_out = grid_out.reshape(2, n_out, 1)

    diffs = grid_in - grid_out
    if periodic:
        periodic_diffs = torch.where(diffs > 0.0, diffs-1, diffs+1)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)


    r = torch.sqrt(diffs[0] ** 2 + diffs[1] ** 2)
    phi = torch.arctan2(diffs[1], diffs[0]) + torch.pi

    idx, vals = kernel_handle(r, phi)
    idx = idx.permute(1, 0)

    return idx, vals

class DiscreteContinuousConv(nn.Module, abc.ABC):
    """
    Abstract base class for DISCO convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, List[int]],
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
    ):
        super().__init__()

        if isinstance(kernel_shape, int):
            self.kernel_shape = [kernel_shape]
        else:
            self.kernel_shape = kernel_shape

        if len(self.kernel_shape) == 1:
            self.kernel_size = self.kernel_shape[0]
        elif len(self.kernel_shape) == 2:
            self.kernel_size = (self.kernel_shape[0] - 1) * self.kernel_shape[1] + 1
        else:
            raise ValueError("kernel_shape should be either one- or two-dimensional.")

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError("Error, the number of input channels has to be an integer multiple of the group size")
        if out_channels % self.groups != 0:
            raise ValueError("Error, the number of output channels has to be an integer multiple of the group size")
        self.groupsize = in_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class EquidistantDiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on arbitrary 2d grids.

    [1] Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, List[int]],
        in_shape: Tuple[int],
        # out_shape: Tuple[int],
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        padding_mode: str = "constant",
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        self.padding_mode = padding_mode

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = 2 * (self.kernel_shape[0]) / float(max(*in_shape))
        self.psi_local_size = math.floor(2*radius_cutoff * max(*in_shape) / 2) + 1

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_size)
        x, y = torch.meshgrid(x, x)
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
        grid_out = torch.Tensor([[0.0], [0.0]])

        idx, vals = _precompute_convolution_tensor_2d(grid_in, grid_out, self.kernel_shape, radius_cutoff=radius_cutoff, periodic=False)

        psi_loc = torch.zeros(self.kernel_size, self.psi_local_size*self.psi_local_size)
        for ie in range(len(vals)):
            f = idx[0, ie]; j = idx[2, ie]; v = vals[ie]
            psi_loc[f, j] = v

        # compute local version of the filter matrix
        psi_loc = psi_loc.reshape(self.kernel_size, self.psi_local_size, self.psi_local_size)
        # normalization by the quadrature weights
        psi_loc = 4.0 * psi_loc / float(in_shape[0]*in_shape[1])

        self.register_buffer("psi_loc", psi_loc, persistent=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        kernel = torch.einsum("kxy,ogk->ogxy", self.psi_loc, self.weight)

        left_pad = self.psi_local_size // 2
        right_pad = (self.psi_local_size+1) // 2 - 1
        x = F.pad(x, (left_pad, right_pad, left_pad, right_pad), mode=self.padding_mode)
        out = F.conv2d(x, kernel, self.bias, stride=1, dilation=1, padding=0, groups=self.groups)

        return out
