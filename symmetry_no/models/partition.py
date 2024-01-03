import numpy as np
import torch
import matplotlib.pyplot as plt




def partition_u(g, M=32, N=32):
    """
    Compute the solution u(x, y) for the 2D Laplacian with given boundary conditions g
    using the normal derivative of the Green's function method.

    Args:
    g: Batched boundary condition function defined on the boundary of the unit square. Shape: (B, n, 4)
    dGdn_func: Function to compute the normal derivative of the Green's function.
    M, N: Number of terms in the series expansion for Green's function approximation.
    grid_size: Size of the discretized grid for the unit square.

    Returns:
    u: Batched approximated solution matrix of size (B, n, n).
    """
    B, n, _ = g.shape
    device = g.device

    # Create the grid indices
    grid = torch.linspace(0, 1, n, device=device)
    x, y = torch.meshgrid(grid, grid, indexing='ij')

    f_left = (1-x) * torch.sin(torch.pi * y)
    f_right = (x) * torch.sin(torch.pi * y)
    f_bottom = torch.sin(torch.pi * x) * (1-y)
    f_top = torch.sin(torch.pi * x) * (y)

    g_left = g[:, :, 0].unsqueeze(2).repeat(1,1,n)
    g_right = g[:, :, 2].unsqueeze(2).repeat(1,1,n)
    g_bottom = g[:, :, 1].unsqueeze(1).repeat(1,n,1)
    g_top = g[:, :, 3].unsqueeze(1).repeat(1,n,1)


    u = f_left * g_left + f_right * g_right + f_bottom * g_bottom + f_top * g_top
    return u


B = 3  # Example batch size
n = 64  # Grid size
g = torch.zeros((B, n, 4))  # Example boundary values
g[:, :, 1] = torch.ones(n)
# g[:, :, 0] = torch.linspace(0,1,n)
u_vectorized = partition_u(g)

# Plot the first image of the solution u
u_to_plot = u_vectorized[0].permute(1,0)  # Select the first batch and ensure it's real
# print(u_to_plot)

plt.figure(figsize=(8, 8))
plt.imshow(u_to_plot, cmap='viridis', origin='lower')
plt.colorbar(label='u(x, y)')
plt.title('Solution of Laplace Equation (First Batch)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()