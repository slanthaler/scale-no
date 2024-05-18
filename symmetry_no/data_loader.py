from pathlib import Path
import os.path
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from symmetry_no.darcy_utilities import DarcyExtractBC
from symmetry_no.helmholtz_utilities import HelmholtzExtractBC
from symmetry_no.data_augmentation import AugmentedTensorDataset, Compose, RandomCropResize, RandomFlip, GridResizing, GridResize
from symmetry_no.rootdir import ROOT_DIR

# taken from 
# [https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python]
def load_mat_v73(filepath):
    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    return arrays

#####################################################################################################
# Darcy Flow Equation
#####################################################################################################


class DarcyReader:
    """
    Helper class to read in Darcy dataset.
    """

    def __init__(self, 
                 mat_file, 
                 root_dir=ROOT_DIR + '/data/', 
                 n_samp=None, 
                 grid_size=None):
        """
        Args:
            mat_file (string): Path to the mat file (Matlab v7.3).
            root_dir (string): Directory with the data.
            n_samp (int): number of samples to extract
            grid_size (int): desired grid size of the output (grid_size==None: keep original)
        """
        self.mat_file = mat_file
        self.root_dir = root_dir
        if self.root_dir:
            self.filepath = root_dir + '/' + mat_file
        else:
            self.filepath = mat_file

        assert os.path.isfile(self.filepath), f'Data file not found! ({self.filepath}).'
        #
        self.n_samp = n_samp
        self.grid_size = grid_size
        
        # load data
        mat = load_mat_v73(self.filepath)
        self.x, self.y = self.unpack_mat(mat, 
                                         n_samp=n_samp, 
                                         grid_size=grid_size)
        
    def __len__(self):
        return len(self.x)

    
    def unpack_mat(self,mat,n_samp=None,grid_size=None):
        """
        Unpack the mat-structure (obtained by reading in a .mat file).
        mat (dict): dictionary from .mat file
        n_samp (int): number of samples to retain
        grid_size (int): map data onto specified grid_size [grid_size==None: keep original]
        """
        # massage the input data
        input_data = torch.tensor(mat['input_data'], dtype=torch.float32).permute(3,2,1,0)
        y = torch.tensor(mat['output_data'], dtype=torch.float32).permute(2,1,0)
        y = y.unsqueeze(1) # add a channel to be consistent with model(x) output
        #
        assert input_data.shape[-1] == input_data.shape[-2], 'Unequal number of grid points along x- and y-direction not supported.' # Nx==Ny
        assert not n_samp or n_samp <= input_data.shape[0], f'Number of requested samples {n_samp} exceeds available number of samples {input_data.shape[0]}.'
        #
        Ns = input_data.shape[-1]
        nchannel = 5 # 1 coefficient + 4 BC
        x = torch.zeros(input_data.shape[0], 
                        nchannel, 
                        input_data.shape[2], 
                        input_data.shape[3], dtype=torch.float32)
        # Filter out boundary conditions (each boundary condition --> 1 channel)
        x[:,0,:,:] = input_data[:,0,:,:]
        x = DarcyExtractBC(x,y)
        print(x.shape)
        #
        del input_data
        
        # 
        if n_samp and n_samp > 0:
            x,y = x[:n_samp], y[:n_samp]

        #
        if grid_size and grid_size>0:
            x = GridResize(x,grid_size)
            y = GridResize(y,grid_size)

        return x, y

class SelfconReader:
    """
    Helper class to read in selfconsistency dataset (this has only input data, no outupts).
    """

    def __init__(self, mat_file, root_dir=ROOT_DIR + '/data/', n_samp=None, grid_size=None):
        """
        Args:
            mat_file (string): Path to the mat file (Matlab v7.3).
            root_dir (string): Directory with the data.
            n_samp (int): number of samples to extract
            grid_size (int): desired grid size of the output (grid_size==None: keep original)
        """
        self.mat_file = mat_file
        self.root_dir = root_dir
        if self.root_dir:
            self.filepath = root_dir + '/' + mat_file
        else:
            self.filepath = mat_file

        assert os.path.isfile(self.filepath), 'Data file not found! ({self.filepath}).'
        #
        self.n_samp = n_samp
        self.grid_size = grid_size
        
        # load data
        mat = load_mat_v73(self.filepath)
        self.x = self.unpack_mat(mat, 
                                 n_samp=n_samp, 
                                 grid_size=grid_size) 
        
    def __len__(self):
        return len(self.x)

    
    def unpack_mat(self,mat,n_samp=None,grid_size=None):
        """
        Unpack the mat-structure (obtained by reading in a .mat file).
        mat (dict): dictionary from .mat file
        n_samp (int): number of samples to retain
        grid_size (int): map data onto specified grid_size [grid_size==None: keep original]
        """
        # massage the input data
        input_data = torch.tensor(mat['input_data'], dtype=torch.float32).permute(3,2,1,0)
        assert input_data.shape[-1] == input_data.shape[-2], 'Unequal number of grid points along x- and y-direction not supported.' # Nx==Ny
        assert not n_samp or n_samp <= input_data.shape[0], f'Number of requested samples {n_samp} exceeds available number of samples {input_data.shape[0]}.'
        #
        Ns = input_data.shape[-1]
        nchannel = 5 # 1 coefficient + 4 BC
        x = torch.zeros(input_data.shape[0], 
                        nchannel, 
                        input_data.shape[2], 
                        input_data.shape[3], dtype=torch.float32)
        # Filter out boundary conditions (each boundary condition --> 1 channel)
        x[:,0,:,:] = input_data[:,0,:,:]
        x = DarcyExtractBC(x,input_data[:,1:,:,:])
        #
        del input_data
        
        # 
        if n_samp and n_samp>0:
            x = x[:n_samp]

        #
        if grid_size and grid_size>0:
            x = GridResize(x,grid_size)

        return x
    
class DarcyData:
    """ 
    Set up train, test and selfconsistency loaders for darcy flow experiment.
    """

    def __init__(self, config):
        # Load training and test datasets
        self.train_file = config.train_data
        if isinstance(config.test_data,list):
            self.test_files = config.test_data
        else:
            self.test_files = [config.test_data]

        if config.selfcon_data:
            self.selfcon = True
            self.selfcon_file = config.selfcon_data
        else:
            self.selfcon = False
            self.selfcon_file = None  

        self.n_train = config.n_train
        self.n_test = config.n_test
        self.grid_size = config.grid_size
        self.batch_size = config.batch_size

        if config.data_dir == None:
            self.root_dir = ROOT_DIR
        else:
            self.root_dir = config.data_dir

        # load datasets
        self.train_data = DarcyReader(self.train_file,
                                      root_dir = self.root_dir,
                                      n_samp=self.n_train,
                                      grid_size=self.grid_size)
        # update the grid_size
        if self.grid_size<0:
            self.grid_size = self.train_data.x.shape[-1]


        self.test_data = []
        for test_file in self.test_files:
            self.test_data.append(DarcyReader(test_file,
                                              root_dir = self.root_dir,
                                              n_samp=self.n_test,
                                              grid_size=self.grid_size))
        if self.selfcon:
            self.selfcon_data = SelfconReader(self.selfcon_file,
                                              root_dir = self. root_dir,
                                              n_samp=self.n_train,
                                              grid_size=self.grid_size)

        if config.use_augmentation:
            cfig = config.use_augmentation.CropResize
            self.t_CropResize = RandomCropResize(p=cfig.p,
                                                 scale_min=cfig.scale_min,
                                                 size_min=cfig.size_min
                                                 )
            self.t_GridResizing = GridResizing(self.grid_size)
            cfig = config.use_augmentation.Flip
            self.t_Flip = RandomFlip(p=cfig.p)
            # compose all of these
            self.transform_xy = Compose([
                self.t_CropResize, 
                self.t_GridResizing, 
                self.t_Flip
                ])
        else:          
            self.transform_xy = None

        # supervised training data
        self.train_db_0 = AugmentedTensorDataset(
            self.train_data.x,
            self.train_data.y,
            transform_xy=self.transform_xy
        )

        # test data
        self.test_dbs = []
        for test_data in self.test_data:
            self.test_dbs.append(
                AugmentedTensorDataset(
                    test_data.x,
                    test_data.y,
                    transform_xy=None
                    )
                )

        # unsupervised training data (if available)
        if self.selfcon:
            self.selfcon_db = torch.utils.data.TensorDataset(
                self.selfcon_data.x
            )
            # stack dataloaders
            self.train_db = torch.utils.data.StackDataset(
                train=self.train_db_0, selfcon=self.selfcon_db
            )
        else:
            self.train_db = torch.utils.data.StackDataset(
                train=self.train_db_0
            )

        # train loader 
        self.train_loader = torch.utils.data.DataLoader(
            self.train_db,
            batch_size=self.batch_size,
            shuffle=True
        )

        # test loader
        self.test_loaders = []
        for test_db in self.test_dbs:
            self.test_loaders.append(
                torch.utils.data.DataLoader(
                    test_db,
                    batch_size=self.batch_size,
                    shuffle=False
                )
            )

#####################################################################################################
# HelmHoltz Equation
#####################################################################################################


class HelmholtzReader:
    """
    Helper class to read in Darcy dataset.
    """

    def __init__(self,
                 mat_file,
                 root_dir=ROOT_DIR + '/data/',
                 n_samp=None,
                 grid_size=None,
                 Re = 1):
        """
        Args:
            mat_file (string): Path to the mat file (Matlab v7.3).
            root_dir (string): Directory with the data.
            n_samp (int): number of samples to extract
            grid_size (int): desired grid size of the output (grid_size==None: keep original)
            Re: wavenumber k
        """
        self.mat_file = mat_file
        self.root_dir = root_dir
        if self.root_dir:
            self.filepath = root_dir + '/' + mat_file
        else:
            self.filepath = mat_file

        assert os.path.isfile(self.filepath), f'Data file not found! ({self.filepath}).'
        #
        self.n_samp = n_samp
        self.grid_size = grid_size

        # load data
        mat = load_mat_v73(self.filepath)
        self.x, self.y = self.unpack_mat(mat,
                                         n_samp=n_samp,
                                         grid_size=grid_size)
        if Re == None:
            Re = 1
        self.re = Re * torch.ones(self.x.shape[0], 1, requires_grad=False)

    def __len__(self):
        return len(self.x)

    def unpack_mat(self, mat, n_samp=None, grid_size=None):
        """
        Unpack the mat-structure (obtained by reading in a .mat file).
        mat (dict): dictionary from .mat file
        n_samp (int): number of samples to retain
        grid_size (int): map data onto specified grid_size [grid_size==None: keep original]
        """
        # massage the input data
        input_data = torch.tensor(mat['input_data'], dtype=torch.float32).permute(3, 2, 1, 0)

        y_real = torch.tensor(mat['output_data']['real'], dtype=torch.float32)
        y_imag = torch.tensor(mat['output_data']['imag'], dtype=torch.float32)
        y = torch.stack([y_real, y_imag], dim=-1).permute(2, 3, 1, 0)

        assert input_data.shape[-1] == input_data.shape[
            -2], 'Unequal number of grid points along x- and y-direction not supported.'  # Nx==Ny
        assert not n_samp or n_samp <= input_data.shape[
            0], f'Number of requested samples {n_samp} exceeds available number of samples {input_data.shape[0]}.'
        #
        Ns = input_data.shape[-1]
        nchannel = 9  # 1 coefficient + 4 BC
        x = torch.zeros(input_data.shape[0],
                        nchannel,
                        input_data.shape[2],
                        input_data.shape[3], dtype=torch.float32)
        # Filter out boundary conditions (each boundary condition --> 1 channel)
        x[:, 0, :, :] = input_data[:, 0, :, :]
        x = HelmholtzExtractBC(x, y)
        # x = DarcyExtractBC(x, x[:, 1, :, :])
        print(x.shape)
        #
        del input_data

        #
        if n_samp and n_samp > 0:
            x, y = x[:n_samp], y[:n_samp]

        #
        if grid_size and grid_size > 0:
            x = GridResize(x, grid_size)
            y = GridResize(y, grid_size)

        return x, y


class HelmholtzData:
    """
    Set up train, test and selfconsistency loaders for darcy flow experiment.
    """

    def __init__(self, config):
        # Load training and test datasets
        if isinstance(config.train_data, list):
            self.train_files = config.train_data
        else:
            self.train_files = [config.train_data]

        if isinstance(config.test_data, list):
            self.test_files = config.test_data
        else:
            self.test_files = [config.test_data]

        if config.selfcon_data:
            self.selfcon = True
            self.selfcon_file = config.selfcon_data
        else:
            self.selfcon = False
            self.selfcon_file = None

        self.n_train = config.n_train
        self.n_test = config.n_test
        self.grid_size = config.grid_size
        self.batch_size = config.batch_size

        if config.data_dir == None:
            self.root_dir = ROOT_DIR
        else:
            self.root_dir = config.data_dir


        self.train_data = []
        for i, train_files in enumerate(self.train_files):
            if config.train_re:
                train_re = config.train_re[i]
            else:
                train_re = None
            self.train_data.append(HelmholtzReader(train_files,
                                              root_dir=self.root_dir,
                                              n_samp=self.n_train,
                                              grid_size=self.grid_size,
                                              Re=train_re))

        self.test_data = []
        for i, test_file in enumerate(self.test_files):
            if config.test_re:
                test_re = config.test_re[i]
            else:
                test_re = None
            self.test_data.append(HelmholtzReader(test_file,
                                              root_dir=self.root_dir,
                                              n_samp=self.n_test,
                                              grid_size=self.grid_size,
                                              Re=test_re))

        if self.selfcon:
            self.selfcon_data = SelfconReader(self.selfcon_file,
                                              root_dir=self.root_dir,
                                              n_samp=self.n_train,
                                              grid_size=self.grid_size)

        if config.use_augmentation:
            cfig = config.use_augmentation.CropResize
            self.t_CropResize = RandomCropResize(p=cfig.p,
                                                 scale_min=cfig.scale_min,
                                                 size_min=cfig.size_min
                                                 )
            self.t_GridResizing = GridResizing(self.grid_size)
            cfig = config.use_augmentation.Flip
            self.t_Flip = RandomFlip(p=cfig.p)
            # compose all of these
            self.transform_xy = Compose([
                self.t_CropResize,
                self.t_GridResizing,
                self.t_Flip
            ])
        else:
            self.transform_xy = None

        # train data
        self.train_dbs = []
        for train_data in self.train_data:
            self.train_dbs.append(
                AugmentedTensorDataset(
                    train_data.x,
                    train_data.y,
                    train_data.re,
                    transform_xy=None
                )
            )

        # test data
        self.test_dbs = []
        for test_data in self.test_data:
            self.test_dbs.append(
                AugmentedTensorDataset(
                    test_data.x,
                    test_data.y,
                    test_data.re,
                    transform_xy=None
                )
            )

        # unsupervised training data (if available)
        # if self.selfcon:
        #     self.selfcon_db = torch.utils.data.TensorDataset(
        #         self.selfcon_data.x
        #     )
        #     # stack dataloaders
        #     self.train_db = torch.utils.data.StackDataset(
        #         train=self.train_db_0, selfcon=self.selfcon_db
        #     )
        # else:
        #     self.train_db = torch.utils.data.StackDataset(
        #         train=self.train_db_0
        #     )

        # test train_loader
        self.train_loaders = []
        for train_db in self.train_dbs:
            self.train_loaders.append(
                torch.utils.data.DataLoader(
                    train_db,
                    batch_size=self.batch_size,
                    shuffle=False
                )
            )
        self.train_loader = self.train_loaders[0]

        # test loader
        self.test_loaders = []
        for test_db in self.test_dbs:
            self.test_loaders.append(
                torch.utils.data.DataLoader(
                    test_db,
                    batch_size=self.batch_size,
                    shuffle=False
                )
            )

#####################################################################################################
# Navier-Stokes Equation
#####################################################################################################

class NSReader:
    """
    Helper class to read in Darcy dataset.
    """

    def __init__(self,
                 mat_file,
                 root_dir=ROOT_DIR + '/data/',
                 N=None,
                 order = "front",
                 truncate=None, T=None, sub_s=None, sub_t=None, T_in=10,
                 Re = 1):
        """
        Args:
            mat_file (string): Path to the mat file (Matlab v7.3).
            root_dir (string): Directory with the data.
            n_samp (int): number of samples to extract
            grid_size (int): desired grid size of the output (grid_size==None: keep original)
        """
        self.mat_file = mat_file
        self.root_dir = root_dir
        self.N = N
        self.T = T
        self.truncate = truncate
        self.sub_t = sub_t
        self.sub_s = sub_s
        self.T_in = T_in
        self.order = order

        if self.root_dir:
            self.filepath = root_dir + '/' + mat_file
        else:
            self.filepath = mat_file

        assert os.path.isfile(self.filepath), f'Data file not found! ({self.filepath}).'

        # load data
        T_out = T + T_in +1
        if order == "front":
            data = torch.load(self.filepath)[:N]
            print(self.filepath, data.shape)
            truncate_S = data.shape[-1] // truncate
            data = data[:, ::sub_t, ::sub_s, ::sub_s][:, :T_out, :truncate_S, :truncate_S]

        else:
            data = torch.load(self.filepath)[-N:]
            print(self.filepath, data.shape)
            truncate_S = data.shape[-1] // truncate
            data = data[:, ::sub_t, ::sub_s, ::sub_s][:, :T_out, :truncate_S, :truncate_S]

        self.x, self.y = self.unpack_mat(data)

        if Re == None:
            Re = 1000
        self.re = Re * torch.ones(self.x.shape[0], 1, requires_grad=False)


        def __len__(self):
            return len(self.x)


    def unpack_mat(self, data, n_samp=None, grid_size=None):
        """

        """
        # massage the input data
        S = data.shape[-1]
        u0 = []
        for i in range(self.T_in):
            ui = data[:, i:self.T+i].reshape(self.N*self.T, 1, S, S)
            u0.append(ui)
        u0 = torch.cat(u0, dim=1)
        u1 = data[:, self.T_in:self.T+self.T_in].reshape(self.N*self.T, S, S)

        x = u0
        y = u1
        print(x.shape, y.shape, torch.mean(torch.abs(x)), torch.mean(torch.abs(y)))
        return x, y

class NSData:
    """
    Set up train, test and selfconsistency loaders for Navier Stokes experiment.
    """

    def __init__(self, config):
        # Load training and test datasets
        self.train_file = config.train_data
        self.T = config.T
        self.truncate = config.truncate
        self.sub_t = config.sub_t
        self.sub_s = config.sub_s
        self.T_in = config.T_in

        if isinstance(config.test_data, list):
            self.test_files = config.test_data
        else:
            self.test_files = [config.test_data]

        if config.selfcon_data:
            self.selfcon = True
            self.selfcon_file = config.selfcon_data
        else:
            self.selfcon = False
            self.selfcon_file = None

        self.n_train = config.n_train
        self.n_test = config.n_test
        self.batch_size = config.batch_size

        if config.data_dir == None:
            self.root_dir = ROOT_DIR
        else:
            self.root_dir = config.data_dir

        if config.train_re:
            train_re = config.train_re
        else:
            train_re = None

        # load datasets
        self.train_data = NSReader(
            mat_file=self.train_file,
            root_dir=self.root_dir,
            N=self.n_train,
            order="front",
            truncate=self.truncate, T=self.T, sub_s=self.sub_s, sub_t=self.sub_t, T_in=self.T_in,
            Re=train_re,
        )
        # update the grid_size
        self.grid_size = self.train_data.x.shape[-1]

        self.test_data = []
        for i, test_file in enumerate(self.test_files):
            if config.test_re:
                test_re = config.test_re[i]
            else:
                test_re = None

            self.test_data.append(
                NSReader(mat_file=test_file,
                         root_dir=self.root_dir,
                         N=self.n_test,
                         order="back",
                         truncate=self.truncate, T=self.T, sub_s=self.sub_s, sub_t=self.sub_t, T_in=self.T_in,
                         Re=test_re,)
            )
        if self.selfcon:
            self.selfcon_data = NSReader(
                mat_file=self.train_file,
                root_dir=self.root_dir,
                N=self.n_train,
                order="front",
                truncate=self.truncate, T=self.T, sub_s=self.sub_s, sub_t=self.sub_t,
                Re=config.selfcon_re,
            )

        if config.use_augmentation:
            cfig = config.use_augmentation.CropResize
            self.t_CropResize = RandomCropResize(p=cfig.p,
                                                 scale_min=cfig.scale_min,
                                                 size_min=cfig.size_min
                                                 )
            self.t_GridResizing = GridResizing(self.grid_size)
            cfig = config.use_augmentation.Flip
            self.t_Flip = RandomFlip(p=cfig.p)
            # compose all of these
            self.transform_xy = Compose([
                self.t_CropResize,
                self.t_GridResizing,
                self.t_Flip
            ])
        else:
            self.transform_xy = None

        # supervised training data
        self.train_db_0 = AugmentedTensorDataset(
            self.train_data.x,
            self.train_data.y,
            self.train_data.re,
            transform_xy=self.transform_xy
        )

        # test data
        self.test_dbs = []
        for test_data in self.test_data:
            self.test_dbs.append(
                AugmentedTensorDataset(
                    test_data.x,
                    test_data.y,
                    test_data.re,
                    transform_xy=None
                )
            )

        # unsupervised training data (if available)
        if self.selfcon:
            self.selfcon_db = torch.utils.data.TensorDataset(
                self.selfcon_data.x
            )
            # stack dataloaders
            self.train_db = torch.utils.data.StackDataset(
                train=self.train_db_0, selfcon=self.selfcon_db
            )
        else:
            self.train_db = torch.utils.data.StackDataset(
                train=self.train_db_0
            )

        # train loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_db,
            batch_size=self.batch_size,
            shuffle=True
        )

        # test loader
        self.test_loaders = []
        for test_db in self.test_dbs:
            self.test_loaders.append(
                torch.utils.data.DataLoader(
                    test_db,
                    batch_size=self.batch_size,
                    shuffle=False
                )
            )
