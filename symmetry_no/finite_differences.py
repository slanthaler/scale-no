
class FiniteDifferencer():
    """
    Computes finite differences etc. for equidistant grid
    """
    def __init__(self,grid0,grid1):
        """
        grid0 - shape (N0,)
        grid1 - shape (N1,)
        """
        grid0,grid1 = torch.meshgrid(grid0,grid1)
        self.grid = torch.stack([grid0,grid1]) # (2,N0,N1) tensor
        self.dx0 = grid0[1] - grid0[0]
        self.dx1 = grid1[1] - grid1[0]

    def __check(self,a):
        """
        Assume that a is of shape (B,Nx,Ny) or (Nx,Ny) with B=batch size
        """
        assert a.ndim>=2, f'Input must be at least two-dimensional. Got {a.ndim} instead.'
        assert a.shape[-2]>1 and a.shape[-1]>1, f'Inputs must have at least two grid points in each direction.'
        
    def Dx0(self,a):
        self.__check(a)
        #
        Dx0a = torch.zeros_like(a)
        Dx0a[...,1:-1,:] = (a[...,2:,:] - a[...,:-2,:]) / (2*self.dx0)
        Dx0a[...,0, :]   = (a[...,1,:] - a[...,0,:]) / self.dx0
        Dx0a[...,-1,:]   = (a[...,-1,:] - a[...,-2,:]) / self.dx0
        return Dx0a

    def Dx1(self,a):
        self.__check(a)
        #
        Dx1a = torch.zeros_like(a)
        Dx1a[...,:,1:-1] = (a[...,:,2:] - a[...,:,:-2]) / (2*self.dx1)
        Dx1a[...,:,0]    = (a[...,:,1] - a[...,:,0]) / self.dx1
        Dx1a[...,:,-1]   = (a[...,:,-1] - a[...,:,-2]) / self.dx1
        return Dx1a

    def rDr(self,a,mid_pt):
        """
        Returns radial derivative about a mid-point mid_pt (shape==(2,)), i.e.
        rDr(a) = (grid-mid_pt)*Dx(a) (vector multiplication on RHS)
        """
        self.__check(a)
        r = self.grid - mid_pt
        return r[0]*self.Dx0(a) + r[1]*self.Dx1(a)
