import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

from symmetry_no.darcy_utilities import DarcyExtractBC

# tensor dataset with augmentation possibility

class AugmentedTensorDataset(Dataset):
    def __init__(self, x, y, re=None, transform_xy=None):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.re = re
        self.transform_xy = transform_xy

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.re is None:
            if self.transform_xy is not None:
                x, y = self.transform_xy(x, y)
            return x, y

        else:
            re = self.re[index]
            if self.transform_xy is not None:
                x,y = self.transform_xy(x,y,re)
        return x,y,re

    def __len__(self):
        return self.x.size(0)


#
def GridResize(x,grid_size,mode='bilinear'):
    """
    Resize the grid values by interpolation in the last two components.
    Expected input is either of size 
        batch x channel x original_size x original_size
    or
        channel x original_size x original_size
    """
    if x.ndim==4:
        return torch.nn.functional.interpolate(x, 
                                               size=(grid_size,grid_size), 
                                               mode='bilinear', 
                                               align_corners=True)
    elif x.ndim==3:
        return torch.nn.functional.interpolate(x.unsqueeze(0), 
                                               size=(grid_size,grid_size), 
                                               mode='bilinear', 
                                               align_corners=True).squeeze(0)
    else:
        ValueError(f'Input x to GridResize must be a tensor with either 3 or 4 dimensions! {x.ndims=}, {x.shape=}')


# data augmentation routines
class GridResizing:
    def __init__(self,grid_size):
        self.grid_size = grid_size

    def __call__(self,x,y):
        return self.forward(x,y)
    
    def forward(self,x,y):
        x = GridResize(x,self.grid_size)
        y = GridResize(y,self.grid_size)
        x = DarcyExtractBC(x,y)
        return x,y

class Compose:
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,x,y):
        return self.forward(x,y)    

    def forward(self,x,y):
        for t in self.transforms:
            x,y=t(x,y)

        return x,y

class RandomFlip:
    def __init__(self,p=0.5, ExtractBC=DarcyExtractBC):
        self.p = p
        self.ExtractBC = ExtractBC

    def __call__(self,x,y):
        return self.forward(x,y)

    def forward(self,x,y):
        if torch.rand(1) < self.p:
            x,y = F.hflip(x), F.hflip(y)
        if torch.rand(1) < self.p:
            x,y = F.vflip(x), F.vflip(y)
        if torch.rand(1) < self.p:
            x,y = x.transpose(-1,-2), y.transpose(-1,-2)
        x = self.ExtractBC(x,y)
        return x,y

class RandomRotation:
    def __init__(self, p=0.5):
        raise NotImplementedError

    def __call__(self,x,y):
        return self.forward(x,y)

    def forward(self,x,y):
        if torch.rand(1) < self.p:
            x,y = x.transpose(-1,-2), y.transpose(-1,-2)
        x = DarcyExtractBC(x,y)
        return x,y

class RandomCropResize:
    def __init__(self, p=0.5, scale_min=0.1, size_min=32):
        """
        Args:
            p (float): probability with which to apply the transformation
            scale_min (float): minimal relative scale of subdomain vs. global domain
            size_min (int): minimal size in terms of grid points
        """
        self.p = p
        self.scale_min = scale_min
        self.size_min = size_min
        self.bbox = None
        #
        assert self.scale_min<=1.0, f'Scaling factor can at most be 1.0, got {scale_min=}'

    def get_params(self, x, y, re=1, rate=None):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            x,y (tensor): Input and corresponding output (assumed having same last two dimensions).
        
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = x.shape[-2], x.shape[-1]
        assert width==height, 'Only allowing width==height for now!.'
        assert width>=self.size_min, f'Cropping only allowed if (width,height)>={self.size_min}. Got {width=}, {height=}'
        #        
        size_min = max(self.size_min,int(round(self.scale_min * width)))
        for _ in range(10):
            if rate == None:
                rnd = torch.rand(1)
                w = size_min + int((width-size_min) * rnd)
                h = size_min + int((height-size_min) * rnd)
            else:
                if rate < 1:
                    rate = 1 / rate
                w = int(max(size_min, width//rate))
                h = int(max(size_min, height//rate))

            # # just for sanity
            # w = min(w,width)
            # h = min(h,height)

            scale =np.sqrt((w/width) * (h/height))
            re = re*scale

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                #
                return i, j, h, w, re

    # def __call__(self,x,y):
    #     return self.forward(x,y)

    def crop(self,x,i,j,h,w):

        return x[...,i:i+h,j:j+w]
    
    # def forward(self,x,y):
    #     if torch.rand(1)>self.p:
    #         return x,y
    #     # take values over random sub-domain
    #     i,j,h,w = self.get_params(x,y)
    #     self.bbox = (i,j,h,w) # only used for illustration
    #     # transform inputs and outputs
    #     x,y  = self.crop(x,i,j,h,w), self.crop(y,i,j,h,w)
    #     #
    #     x = DarcyExtractBC(x,y)
    #
    #     return x,y


class RandomCropResizeTime:
    def __init__(self, p=0.5, scale_min=0.1, size_min=32):
        """
        Args:
            p (float): probability with which to apply the transformation
            scale_min (float): minimal relative scale of subdomain vs. global domain
            size_min (int): minimal size in terms of grid points
        """
        self.p = p
        self.scale_min = scale_min
        self.size_min = size_min
        self.bbox = None
        #
        assert self.scale_min <= 1.0, f'Scaling factor can at most be 1.0, got {scale_min=}'

    def get_params(self, x, y, re=1, rate=None):

        """Get parameters for ``crop`` for a random sized crop.

        Args:
            x,y (tensor): Input and corresponding output (assumed having same last two dimensions).

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        T, S = x.shape[-2], x.shape[-1]
        assert T >= self.size_min, f'Cropping only allowed if (width,height)>={self.size_min}. Got {width=}, {height=}'

        size_min = max(self.size_min, int(round(self.scale_min * T)))

        if rate == None:
            rnd = torch.rand(1)
            t = size_min + int((T - size_min) * rnd)
            rate = T/t
        else:
            if rate < 1:
                rate = 1/rate

        t = int(max(size_min, T // rate))
        s = int(max(size_min, S // rate))

        scale = (s / S)
        re = re * scale

        i = torch.randint(0, T - t + 1, size=(1,)).item()
        j = torch.randint(0, S - s + 1, size=(1,)).item()

        return i, j, t, s, re

    def crop(self, x, i, j, h, w):
        return x[..., i:i + h, j:j + w]

class RandomCropResizeTimeAR:
    def __init__(self, p=0.5, scale_min=0.1, size_min=32):
        """
        Args:
            p (float): probability with which to apply the transformation
            scale_min (float): minimal relative scale of subdomain vs. global domain
            size_min (int): minimal size in terms of grid points
        """
        self.p = p
        self.scale_min = scale_min
        self.size_min = size_min
        self.bbox = None
        #
        assert self.scale_min <= 1.0, f'Scaling factor can at most be 1.0, got {scale_min=}'

    def get_params(self, x, y, re=1, rate=None):

        """Get parameters for ``crop`` for a random sized crop.

        Args:
            x,y (tensor): Input and corresponding output (assumed having same last two dimensions).

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """

        width, height = x.shape[-2], x.shape[-1]
        assert width==height, 'Only allowing width==height for now!.'
        assert width>=self.size_min, f'Cropping only allowed if (width,height)>={self.size_min}. Got {width=}, {height=}'
        #
        size_min = max(self.size_min,int(round(self.scale_min * width)))
        for _ in range(10):
            if rate == None:
                rnd = torch.rand(1)
                w = size_min + int((width-size_min) * rnd)
                h = size_min + int((height-size_min) * rnd)
            else:
                if rate < 1:
                    rate = 1 / rate
                w = int(max(size_min, width//rate))
                h = int(max(size_min, height//rate))

            scale = np.sqrt((w/width) * (h/height))
            re = re * scale**2

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                #
                return i, j, h, w, re

    def crop(self, x, i, j, h, w):
        x = x[..., i:i + h, j:j + w]
        return x

