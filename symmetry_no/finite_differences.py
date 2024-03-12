import torch 

class FiniteDifferencer():
    """
    Computes finite differences etc. for equidistant grid
    """
    def __init__(self,grid0,grid1):
        """
        grid0 - shape (N0,)
        grid1 - shape (N1,)
        """
        self.dx0 = grid0[1] - grid0[0]
        self.dx1 = grid1[1] - grid1[0]
        # create 2d grid
        grid0,grid1 = torch.meshgrid(grid0,grid1)
        self.grid = torch.stack([grid0,grid1]) # (2,N0,N1) tensor

    def __check(self,a):
        """
        Assume that a is of shape (B,Nx,Ny) or (Nx,Ny) with B=batch size
        """
        assert a.ndim>=2, f'Input must be at least two-dimensional. Got {a.ndim} instead.'
        assert a.shape[-2]>1 and a.shape[-1]>1, f'Inputs must have at least two grid points in each direction.'
        
    def Dx0(self,a):
        self.__check(a)
        #
        Dx0a = torch.zeros_like(a, device=a.device)
        Dx0a[...,1:-1,:] = (a[...,2:,:] - a[...,:-2,:]) / (2*self.dx0)
        #Dx0a[...,0, :]   = (a[...,1,:] - a[...,0,:]) / self.dx0
        #Dx0a[...,-1,:]   = (a[...,-1,:] - a[...,-2,:]) / self.dx0
        # the following is a higher-order version at boundary (2nd order accurate)
        Dx0a[...,0,:] = (4*a[...,1,:] - 3*a[...,0,:] - a[...,2,:])/(2*self.dx0)
        Dx0a[...,-1,:] = -(4*a[...,-2,:] - 3*a[...,-1,:] - a[...,-3,:])/(2*self.dx0)
        return Dx0a

    def Dx1(self,a):
        self.__check(a)
        #
        Dx1a = torch.zeros_like(a, device=a.device)
        Dx1a[...,:,1:-1] = (a[...,:,2:] - a[...,:,:-2]) / (2*self.dx1)
        #Dx1a[...,:,0]    = (a[...,:,1] - a[...,:,0]) / self.dx1
        #Dx1a[...,:,-1]   = (a[...,:,-1] - a[...,:,-2]) / self.dx1
        # the following is a higher-order version at boundary  (2nd order accurate)
        Dx1a[...,:,0] = (4*a[...,:,1] - 3*a[...,:,0] - a[...,:,2])/(2*self.dx1)
        Dx1a[...,:,-1] = -(4*a[...,:,-2] - 3*a[...,:,-1] - a[...,:,-3])/(2*self.dx1)
        return Dx1a

    def rDr(self,a,mid_pt):
        """
        Returns radial derivative about a mid-point mid_pt (shape==(2,)), i.e.
        rDr(a) = (grid-mid_pt)*Dx(a) (vector multiplication on RHS)
        """
        self.__check(a)
        r = self.grid - mid_pt.view(2,1,1)
        r = r.to(a.device)
        return r[0]*self.Dx0(a) + r[1]*self.Dx1(a)



if __name__=='__main__':

    # some basic tests for finite differencer
    N0,N1 = 10,10
    grid0,grid1 = torch.linspace(0,1,N0), torch.linspace(0,1,N1)
    FD = FiniteDifferencer(grid0,grid1)
    #
    a = torch.rand((2,N0,N1))
    print(FD.Dx0(a).shape, FD.Dx1(a).shape, FD.rDr(a,torch.rand((2,1,1))).shape)
    
    # check rDr(x1) == x1, rDr(x2) == x2
    a = FD.grid
    fd = FD.rDr(a,torch.zeros((2,)))
    print('rDr on [x1,x2] gives correct result? ',torch.allclose(a,fd))

    # check correct derivatives of exp(sin(k0*x0) + sin(k1*x1))
    k0,k1 = 5,10
    N = 100
    grid = torch.linspace(0,1,N)
    FD = FiniteDifferencer(grid,grid)
    #
    x0,x1 = torch.meshgrid(grid,grid)
    a = torch.exp(torch.sin(k0*x0) + torch.sin(k1*x1))

    # analytic derivatives
    da_dx0 = k0*torch.cos(k0*x0) * a
    da_dx1 = k1*torch.cos(k1*x1) * a

    # numerical approximation
    ax0 = FD.Dx0(a)
    ax1 = FD.Dx1(a)

    # I believe these are correct, but the derivatives at the boundary are only 1st order approximations and therefore quite inaccurate(?) in float precision, especially(?)

    print('a = exp(sin(k0*x0)+sin(k1*x1))')
    print('da/dx0 correct? ',torch.allclose(da_dx0[1:-1,1:-1],ax0[1:-1,1:-1],rtol=1e-2))
    print('da/dx1 correct? ',torch.allclose(da_dx1[1:-1,1:-1],ax1[1:-1,1:-1],rtol=1e-2))

    print('mean / max: ',
          torch.abs(da_dx0-ax0).mean(),
          torch.abs(da_dx0-ax0).max())
    print('index: ',torch.argmax(torch.abs(da_dx0-ax0)))
    print(da_dx0.view(-1)[-1]-ax0.view(-1)[-1])
    
#    import matplotlib.pyplot as plt
#    fig,axs=plt.subplots(1,2,figsize=(6,3))
#    im = axs[0].pcolor(da_dx0)
#    fig.colorbar(im,ax=axs[0])
#    im = axs[1].pcolor(torch.log(torch.abs(ax0-da_dx0)))
#    fig.colorbar(im,ax=axs[1])
#    plt.show()
