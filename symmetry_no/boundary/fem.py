import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import time

# Define the grid size
N = 50
h = 1.0 / (N - 1)

class LaplacianSolver:
    def __init__(self, N, device="cpu"):
        self.N = N
        self.h = 1.0 / (N - 1)
        self.device = device
        self.K = torch.zeros((1, N * N, N * N), device=self.device)
        self.assemble_stiffness_matrix()

    def idx(self, i, j):
        return i * self.N + j

    def assemble_stiffness_matrix(self):
        N = self.N
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                self.K[:, self.idx(i, j), self.idx(i, j)] = 4
                self.K[:, self.idx(i, j), self.idx(i - 1, j)] = -1
                self.K[:, self.idx(i, j), self.idx(i + 1, j)] = -1
                self.K[:, self.idx(i, j), self.idx(i, j - 1)] = -1
                self.K[:, self.idx(i, j), self.idx(i, j + 1)] = -1

            for i in range(N):
                # Top boundary
                self.K[:, self.idx(0, i), :] = 0
                self.K[:, self.idx(0, i), self.idx(0, i)] = 1

                # Bottom boundary
                self.K[:, self.idx(N - 1, i), :] = 0
                self.K[:, self.idx(N - 1, i), self.idx(N - 1, i)] = 1

                # Left boundary
                self.K[:, self.idx(i, 0), :] = 0
                self.K[:, self.idx(i, 0), self.idx(i, 0)] = 1

                # Right boundary
                self.K[:, self.idx(i, N - 1), :] = 0
                self.K[:, self.idx(i, N - 1), self.idx(i, N - 1)] = 1

    def apply_boundary_conditions(self, g):
        batchsize = g.shape[0]
        N = self.N
        F = torch.zeros(batchsize, N * N, device=self.device)  # Reset F for new boundary conditions

        for i in range(N):
            # Top boundary
            F[:, self.idx(0, i)] = g[:, i, 0]
            # Bottom boundary
            F[:, self.idx(N - 1, i)] = g[:, i, 2]
            # Left boundary
            F[:, self.idx(i, 0)] = g[:, i, 3]
            # Right boundary
            F[:, self.idx(i, N - 1)] = g[:, i, 1]
        return F

    def solve(self, g):
        F = self.apply_boundary_conditions(g)
        F = F.unsqueeze(-1)
        K = self.K.repeat(F.shape[0], 1 ,1)
        u_flat = torch.linalg.solve(K, F)
        # L = torch.linalg.cholesky(K)
        # u_flat = torch.cholesky_solve(F, L)
        u = u_flat.view(-1, self.N, self.N)
        return u


N = 128
FEM = LaplacianSolver(N, device="cuda")
g = torch.zeros(10, N, 4)
g[:, :, 3] = torch.sin(2*torch.pi*4*torch.linspace(1,0,N))
g[:, :, 0] = torch.sin(2*torch.pi*4*torch.linspace(0,1,N))
t1 = time.time()
u = FEM.solve(g)
t2 = time.time()
u = u.cpu()
print(t2 - t1)

# Plot the first image of the solution u
u_to_plot = u[0]  # Select the first batch and ensure it's real
# print(u_to_plot)

plt.figure(figsize=(8, 8))
plt.imshow(u_to_plot, cmap='viridis', origin='lower')
plt.colorbar(label='u(x, y)')
plt.title('Solution of Laplace Equation (First Batch)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
