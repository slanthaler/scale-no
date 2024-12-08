import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

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
    for m in range(1, M + 1):
        for n in range(1, N + 1):
            term = (
                4 * np.sin(m * np.pi * xi) * np.sin(n * np.pi * eta) *
                np.sin(m * np.pi * x) * np.sin(n * np.pi * y) /
                (np.pi**2 * (m**2 + n**2))
            )
            G += term
    return G

def dGdn(x, y, xi, eta, M=4, N=4, direction="top"):
    """
    Calculate the normal derivative of the Green's function for the 2D Laplacian on a unit square
    with Dirichlet boundary conditions for different directions.

    Args:
    x, y: Coordinates of the point where the normal derivative of Green's function is evaluated.
    xi, eta: Coordinates of the source point on the boundary.
    M, N: Number of terms in the series expansion for approximation.
    direction: The normal direction, can be "x", "-x", "y", "-y".

    Returns:
    dGdn: Value of the normal derivative of the Green's function at point (x, y) due to source at (xi, eta).
    """
    dGdn = 0

    # left (0, y)
    if direction == "left":
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                dGdn += (4 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y) *
                         m * np.sin(n * np.pi * eta) /
                         (np.pi * (m**2 + n**2)))

    # right (1, y)
    elif direction == "right":
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                dGdn += (4 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y) *
                         m * np.sin(n * np.pi * eta) /
                         (np.pi * (m**2 + n**2)))

    # bottom (x, 0)
    elif direction == "bottom":
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                dGdn += (4 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y) *
                         np.sin(m * np.pi * xi) * n /
                         (np.pi * (m**2 + n**2)))

    # top (x, 1)
    elif direction == "top":
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                dGdn += (4 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y) *
                         np.sin(m * np.pi * xi) * n /
                         (np.pi * (m**2 + n**2)))

    return dGdn



print(dGdn(0.1,0.5, 0, 0.5, M=2, N=1, direction="left"))
print(dGdn(0.9,0.5, 1, 0.5, M=2, N=1, direction="right"))

def compute_solution_u(g, M=1, N=1):
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

    u = torch.zeros((B, n, n), device=device)
    dx = dy = 1 / n  # Grid spacing

    for i in range(n):
        for j in range(n):
            x, y = i * dx, j * dy  # Grid point coordinates
            for k in range(n):
                xi, eta = i * dx, k * dy
                # Accumulate contributions from the boundary
                u[:, i, j] += dGdn(x, y, xi, 1, M, N, "top") * g[:, k, 3] * dy  # top
                u[:, i, j] += dGdn(x, y, xi, 0, M, N, "bottom") * g[:, k, 1] * dy  # bottom
                u[:, i, j] += dGdn(x, y, 0, eta, M, N, "left") * g[:, k, 0] * dx  # left
                u[:, i, j] += dGdn(x, y, 1, eta, M, N, "right") * g[:, k, 2] * dx  # right

    return u


B = 3  # Example batch size
n = 20  # Grid size
g = torch.zeros((B, n, 4))  # Example boundary values
# g[:, :, 3] = torch.linspace(1,0,n)
g[:, :, 2] = torch.ones(n)
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
