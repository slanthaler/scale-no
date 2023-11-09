import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

from darcy_utilities import DarcyExtractBC

# tensor dataset with augmentation possibility

class AugmentedTensorDataset(Dataset):
    def __init__(self, x, y, transform_xy=None):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.transform_xy = transform_xy

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform_xy is not None:
            x,y = self.transform_xy(x,y)

        return x,y

    def __len__(self):
        return self.x.size(0)



# data augmentation routines

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
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,x,y):
        return self.forward(x,y)

    def forward(self,x,y):
        if torch.rand(1) < self.p:
            rnd = torch.rand(1)
            if rnd < 0.25:
                x,y = F.hflip(x), F.hflip(y)
            elif rnd < 0.5:
                x,y = F.vflip(x), F.vflip(y)
            elif rnd < 0.75:
                x,y, = F.vflip(F.hflip(x)), F.vflip(F.hflip(y))        
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

    def get_params(self, x, y):
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
            rnd = torch.rand(1)
            w = size_min + int((width-size_min) * rnd)
            h = size_min + int((height-size_min) * rnd)

            # just for sanity
            w = min(w,width)
            h = min(h,height)

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                #
                return i, j, h, w

    def __call__(self,x,y):
        return self.forward(x,y)

    def crop(self,x,i,j,h,w):
        return x[...,i:i+h,j:j+w].clone()
    
    def forward(self,x,y):
        if torch.rand(1)>self.p:
            return x,y
        # take values over random sub-domain
        i,j,h,w = self.get_params(x,y)
        self.bbox = (i,j,h,w) # only used for illustration
        # transform inputs and outputs
        print('size before crop, ', x.shape)
        x,y  = self.crop(x,i,j,h,w), self.crop(y,i,j,h,w)
        # 
        print('size after crop, ', x.shape)
        x = DarcyExtractBC(x,y)
        return x,y

class RandomRotation:
    def __init__(self, p=0.5):
        raise NotImplementedError

    def __call__(self,x,y):
        return self.forward(x,y)

    def forward(self,x,y):
        pass
