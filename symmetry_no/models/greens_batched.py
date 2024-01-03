import numpy as np
import torch
import matplotlib.pyplot as plt

def greens_function_2d_laplacian(x, y, xi, eta, M=10, N=10):
    """
    Calculate the Green's function for the 2D Laplacian on a unit square
    with Dirichlet boundary conditions.

    Args:
    x, y: Coordinates of the point where Green's function is evaluated.
    xi, eta: Coordinates of the source point.
    M, N: Number of terms in the series expansion for approximation.

    Returns:
    G: Value of the Green's function at point (x, y) due to source at (xi, eta).
    """
    G = 0
    device = x.device
    m = torch.arange(1,M+1,device=device)
    n = torch.arange(1,N+1,device=device)
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            term = (
                4 * np.sin(m * np.pi * xi) * np.sin(n * np.pi * eta) *
                np.sin(m * np.pi * x) * np.sin(n * np.pi * y) /
                (np.pi**2 * m * n * (m**2 + n**2))
            )
            G += term
    return G

def dGdn(x, y, xi, eta, M=4, N=4):
    """
    Calculate the normal derivative of the Green's function for the 2D Laplacian on a unit square
    with Dirichlet boundary conditions for different directions.

    Args:
    x, y: Coordinates of the point where the normal derivative of Green's function is evaluated.
    xi, eta: Coordinates of the source point on the boundary.
    M, N: Number of terms in the series expansion for approximation.
    direction: assumed for left

    Returns:
    dGdn: Value of the normal derivative of the Green's function at point (x, y) due to source at (xi, eta).
    """
    dGdn = 0
    # left (0, y)
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            dGdn += (4 * torch.sin(m * torch.pi * x) * torch.sin(n * torch.pi * y) *
                     m * torch.sin(n * torch.pi * eta) /
                     (torch.pi * (m**2 + n**2)))
    return dGdn


def compute_solution_u(g, M=64, N=64):
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
    grid = torch.linspace(1/(2*n), (2*n-1)/(2*n), n, device=device)
    x, y, eta = torch.meshgrid(grid,
                               grid,
                               grid, indexing='ij')


    G_left = dGdn(x, y, 0, eta, M, N)
    G_right = G_left.flip(0)
    G_bottom = G_left.permute(1,0,2)
    G_top = G_bottom.flip(1)


    g_left = g[:, :, 0].unsqueeze(1).unsqueeze(2).repeat(1,n,n,1)
    g_right = g[:, :, 2].unsqueeze(1).unsqueeze(2).repeat(1,n,n,1)
    g_bottom = g[:, :, 1].unsqueeze(1).unsqueeze(2).repeat(1,n,n,1)
    g_top = g[:, :, 3].unsqueeze(1).unsqueeze(2).repeat(1,n,n,1)

    u = G_top * g_top + G_bottom * g_bottom + G_left * g_left + G_right * g_right
    u = u.mean(-1)
    u[:,0,:] = g[:, :, 0]
    u[:,-1,:] = g[:, :, 2]
    u[:,:,0] = g[:, :, 1]
    u[:,:,-1] = g[:, :, 3]

    return u


B = 3  # Example batch size
n = 64  # Grid size
g = torch.zeros((B, n, 4))  # Example boundary values
g[:, :, 3] = torch.linspace(1,0,n)
g[:, :, 0] = torch.linspace(0,1,n)
# g = torch.ones((B, n, 4))
u_vectorized = compute_solution_u(g)

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