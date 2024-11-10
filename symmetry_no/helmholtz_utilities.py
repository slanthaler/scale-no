import matplotlib.pyplot as plt


def FD_x(a, dx):
    """
    Finite difference in x-direction.
    """
    return (a[..., 1:, :] - a[..., :-1, :]) / dx


def FD_y(a, dy):
    """
    Finite difference in y-direction.
    """
    return (a[..., :, 1:] - a[..., :, :-1]) / dy


def AV_x(a):
    """
    Averaging in x-direction.
    """
    return (a[..., 1:, :] + a[..., :-1, :]) / 2.


def AV_y(a):
    """
    Averaging in y-direction.
    """
    return (a[..., :, 1:] + a[..., :, :-1]) / 2.


def PDEResidual(x, y):

    return None


def PlotResidual(x, y, subsamp=8):
    """
    Plot pointwise residual (Darcy PDE).
    """
    res = PDEResidual(x, y)

    if len(x.shape) < 4:
        NotImplementedError(f'PlotResidual: {x.shape=} is not supported. Need 4 indices!')

    for i in range(x.shape[0]):
        a = x[i, 0, ::subsamp, ::subsamp].squeeze().detach().numpy()
        u = y[i, 0, ::subsamp, ::subsamp].squeeze().detach().numpy()
        resi = res[i, ::subsamp, ::subsamp].squeeze().detach().numpy()
        #
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        # plot a
        pc0 = axs[0].pcolor(a.T)
        fig.colorbar(pc0, ax=axs[0])
        axs[0].set_title('a(x)')

        # plot u
        pc1 = axs[1].pcolor(u.T)
        fig.colorbar(pc1, ax=axs[1])
        axs[1].set_title('u(x)')

        # plot residual
        pc2 = axs[2].pcolor(resi.T)
        fig.colorbar(pc2, ax=axs[2])
        axs[2].set_title('residual(x)')


def HelmholtzExtractBC(x, y):
    """
    Extract boundary conditions from y and add them to channels 1,...,4 in x.
    """
    unsqueeze_x = (x.ndim == 3)  # check whether batch dimension is missing
    unsqueeze_y = (y.ndim == 3)

    # add batch dimension
    if unsqueeze_x:
        x = x.unsqueeze(0)
    if unsqueeze_y:
        y = y.unsqueeze(0)

    # extract BC
    Nx, Ny = y.shape[-2], y.shape[-1]
    x[:, 1:3, :, :] = y[:, :, 0, :].unsqueeze(-2).repeat(1, 1, Nx, 1)  # left boundary
    x[:, 3:5, :, :] = y[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, Ny)  # lower boundary
    x[:, 5:7, :, :] = y[:, :, -1, :].unsqueeze(-2).repeat(1, 1, Nx, 1)  # right boundary
    x[:, 7:9, :, :] = y[:, :, :, -1].unsqueeze(-1).repeat(1, 1, 1, Ny)  # upper boundary

    # undo adding of batch dimension
    if unsqueeze_x:
        x = x[0]
    if unsqueeze_y:
        y = y[0]

    return x

