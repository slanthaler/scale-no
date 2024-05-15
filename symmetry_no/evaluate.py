import sys
import wandb
import argparse

# sys.path.append("/central/groups/astuart/zongyi/symmetry-no/")

from symmetry_no.data_loader import DarcyData, HelmholtzData, NSData
from symmetry_no.config_helper import ReadConfig
from symmetry_no.wandb_utilities import *
from symmetry_no.models.fno2d import *
from symmetry_no.models.fno2d_doubled import *
from symmetry_no.models.fno_u import *
from symmetry_no.models.fno_re import *
from symmetry_no.selfconsistency import LossSelfconsistency
from symmetry_no.gaussian_random_field import sample_NS

def main(config):
    #
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using cuda? {device}')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # load training and test datasets
    if config.dataset == 'darcy':
        print('Loading Darcy datasets...')
        data = DarcyData(config)
        n_train = config.n_train
        n_test = config.n_test

    elif config.dataset == 'helmholtz':
        print('Loading Helmholtz datasets...')
        data = HelmholtzData(config)
        n_train = config.n_train
        n_test = config.n_test

    elif config.dataset == 'NS':
        print('Loading NS datasets...')
        data = NSData(config)
        n_train = config.n_train * config.T
        n_test = config.n_test * config.T

    else:
        print("config.dataset should be either 'darcy' or 'NS'.")

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    modes1 = config.modes
    modes2 = config.modes
    width = config.width
    depth = config.depth
    mlp = config.mlp

    ### U-shape FNO
    S = config.S
    modes = S//2
    modes_list = []
    width_list = []
    for i in range(depth):
        n = 2**i
        modes_list.append(modes//n)
        width_list.append(n*width)

    # model = FNO2d(modes1, modes2, width, depth).to(device)
    # model = FNO2d_doubled(modes1, modes2, width, depth).to(device)
    # model = FNO_U(modes_list, modes_list, width_list, depth=3, mlp=mlp, in_channel=7, out_channel=1).to(device)
    model = FNO_mlp(width, modes1, modes2, depth, mlp=mlp, in_channel=7, out_channel=1).to(device)
    print('FNO2d parameter count: ', count_params(model))

    PATH = '/home/wumming/Documents/symmetry-no/symmetry_no/model_mlp_0508.h5'
    model.load_state_dict(torch.load(PATH))
    model.eval()

    #
    batch_size = config.batch_size
    loss_fn = LpLoss(size_average=False)

        #    t1 = default_timer()
    # test_l2 = np.zeros((len(data.test_loaders),))
    # test_rmse = np.zeros((len(data.test_loaders),))
    # test_identity = np.zeros((len(data.test_loaders),))
    #
    #
    #
    # # per-step error
    # with torch.no_grad():
    #     for i, test_loader in enumerate(data.test_loaders):
    #         for x, y, re in test_loader:
    #             x, y, re = x.to(device), y.to(device), re.to(device)
    #
    #             out = model(x, re)
    #             out = out.squeeze()
    #             y = y.squeeze()
    #             # out = x[:,0].reshape(batch_size, -1)
    #             # if i == 5:
    #             #     x = x[..., ::4, ::4]
    #             #     y = y[..., ::4, ::4]
    #             #     out = out[..., ::4, ::4]
    #
    #             test_l2[i] += loss_fn(out, y).item()
    #             test_rmse[i] += loss_fn.abs(out, y).item()
    #             test_identity[i] += loss_fn.i_rel(out, y, x[:,0]).item()
    #
    # # normalize losses
    # test_l2 /= n_test
    # test_rmse /= n_test
    # test_identity /= n_test
    #
    #
    # test_losses = " / ".join([f"{val:.5f}" for val in test_l2])
    # test_rmse_losses = " / ".join([f"{val:.5f}" for val in test_rmse])
    # test_identity_losses = " / ".join([f"{val:.5f}" for val in test_identity])
    # print(f'test: {test_losses}, \n'
    #       f'test: {test_rmse_losses}, \n'
    #       f'test: {test_identity_losses}, \n',
    #     flush=True)

    # trajectory

    with torch.no_grad():
        for i, test_loader in enumerate(data.test_loaders):
            for x, y, re in test_loader:
                x, y, re = x.to(device), y.to(device), re.to(device)

                out = model(x, re)
                out = out.squeeze(1)
                # out = x[:,0].reshape(batch_size, -1)
                # if i == 5:
                #     x = x[..., ::4, ::4]
                #     y = y[..., ::4, ::4]
                #     out = out[..., ::4, ::4]

                test_l2[i] += loss_fn(out, y).item()
                test_rmse[i] += loss_fn.abs(out, y).item()
                test_identity[i] += loss_fn.i_rel(out, y, x[:,0]).item()
#
if __name__ == '__main__':
    # parse command line arguments
    # (need to specify <name> of run = config_<name>.yaml)
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-n', "--name",
                        type=str,
                        default='ns',
                        help="Specify name of run (requires: config_<name>.yaml in ./config folder).")
    parser.add_argument('-c', "--config",
                        type=str,
                        help="Specify the full config-file path.")
    args = parser.parse_args()

    # read the config file
    config = ReadConfig(args.name, args.config)

    #
    print('Command line inputs: --')
    print('Config name: ', args.name)
    print('Config file: ', args.config, flush=True)

    # run the main training loop
    main(config)
