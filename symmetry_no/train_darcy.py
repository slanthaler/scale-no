import sys
import wandb
import argparse

# sys.path.append("/central/groups/astuart/zongyi/symmetry-no/")
#
from symmetry_no.data_loader import DarcyData, HelmholtzData, NSData
from symmetry_no.config_helper import ReadConfig
from symmetry_no.wandb_utilities import *
from symmetry_no.models.fno2d import *
from symmetry_no.models.fno2d_doubled import *
from symmetry_no.models.fno_u import *
from symmetry_no.models.fno_re import *
from symmetry_no.models.unet import UNet2d
sys.path.append("/pscratch/sd/z/zongyili/mhchen/neuraloperator")
from neuralop.models.uno import UNO
from symmetry_no.selfconsistency import LossSelfconsistency
from symmetry_no.super_sample import sample_Darcy
from symmetry_no.data_augmentation import RandomFlip

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
        in_channel = 5

    elif config.dataset == 'helmholtz':
        print('Loading Helmholtz datasets...')
        data = HelmholtzData(config)
        n_train = config.n_train
        n_test = config.n_test
        in_channel = 9

    elif config.dataset == 'NS':
        print('Loading NS datasets...')
        data = NSData(config)
        n_train = config.n_train * config.T
        n_test = config.n_test * config.T
        in_channel = config.T_in

    else:
        print("config.dataset should be either 'darcy' or 'NS'.")

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)

    width = config.width
    modes = config.modes if hasattr(config, 'modes') else None
    depth = config.depth if hasattr(config, 'depth') else None

    if config.model == "UNet":
        model = UNet2d(in_dim=in_channel, out_dim=1, latent_size=config.grid_size).to(device)
    elif config.model == "FNO":
        model = FNO2d(modes, modes, width, depth, in_channel=in_channel, boundary=True).to(device)
    elif config.model == "FNO_u":
        modes1_list = [modes] * config.level
        modes2_list = [modes] * config.level
        width_list = [width] * config.level
        model = FNO_U(modes1_list, modes2_list, width_list, 
                     level=config.level, 
                     mlp=False,
                     depth=depth, 
                     in_channel=in_channel,
                     scale_informed=True,
                     frequency_pos_emb=True,
                     boundary=False).to(device)
    elif config.model == "UNO":
        model = UNO(
            in_channels=in_channel,
            out_channels=1,
            hidden_channels=width,
            uno_out_channels=config.uno_out_channels,
            uno_n_modes=config.uno_n_modes,
            uno_scalings=config.uno_scalings,
            lifting_channels=width,
            projection_channels=width,
            channel_mlp_skip="linear",
            fno_skip="linear"
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {config.model}. Supported models are: UNet, FNO, FNO_u, UNO")

    batch_size = config.batch_size
    epochs = config.epochs
    epoch_test = config.epoch_test
    start_selfcon = config.epochs_selfcon
    track_selfcon = config.track_selfcon
    augmentation_samples = config.augmentation_samples
    sample_virtual_instance = config.sample_virtual_instance

    use_augmentation = config.augmentation_loss
    iterations = epochs * max(1, n_train // batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    if args.wandb:
        wandb.watch(model, log_freq=20, log="all")

    loss_fn = LpLoss(size_average=False)
    loss_mse = nn.MSELoss(reduction='sum')
    group_action = RandomFlip()
    for epoch in range(0, epochs + 1):  # config.epochs
        #
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_aug = 0
        train_sc = 0

        # training loop
        for d in data.train_loader:
            x, y = d
            x, y = x.to(device), y.to(device)
            # x, y = group_action(x, y)

            # supervised training
            optimizer.zero_grad()
            if config.model == "UNO":
                out = model(x)
            else:
                out = model(x, re=None)
            # DO WE NEED TO NORMALIZE THE OUTPUT??

            loss = loss_fn(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            train_l2 += loss.item()

            # augmentation via sub-sampling
            for j in range(augmentation_samples):
                if use_augmentation:
                    loss_aug = LossSelfconsistency(model, x, loss_fn, y=y, group_action=group_action)
                    loss += 1.0 * loss_aug
                    train_aug += loss_aug.item()

                if sample_virtual_instance and (epoch >= start_selfcon):
                    rate = torch.rand(1) * 3 + 1
                    new_x, rate = sample_Darcy(input=x, rate=rate, keepsize=True)
                    loss_sc = LossSelfconsistency(model, new_x, loss_fn)
                    loss += 0.25 * loss_sc * (epoch / epochs)
                    train_sc += loss_sc.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        #
        model.eval()
        test_l2 = np.zeros((len(data.test_loaders),))
        if epoch % epoch_test == 0:
            with torch.no_grad():
                for i, test_loader in enumerate(data.test_loaders):
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)

                        if config.model == "UNO":
                            out = model(x)
                        else:
                            out = model(x, re=None)
                        # DO WE NEED TO NORMALIZE THE OUTPUT??
                        test_l2[i] += loss_fn(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        # normalize losses
        train_l2 /= (n_train )
        train_sc /= (n_train * augmentation_samples)
        train_aug /= (n_train * augmentation_samples)
        test_l2 /= n_test

        t2 = default_timer()
        if args.wandb:
            wandb.log({'time': t2 - t1, 'train_l2': train_l2, 'train_selfcon': train_sc})
            wandb.log({f"test_l2/loss-{ii}": loss for ii, loss in enumerate(test_l2)})
        test_losses = " / ".join([f"{val:.5f}" for val in test_l2])
        print(
            f'[{epoch:3}], time: {t2 - t1:.3f}, train: {train_l2:.5f}, test: {test_losses}, train_aug: {train_aug:.5f}, train_sc: {train_sc:.5f}',
            flush=True)

    #    # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    if args.wandb:
        torch.save(model.state_dict(), "model.h5")
    wandb.save('model.h5')


#
if __name__ == '__main__':
    # parse command line arguments
    # (need to specify <name> of run = config_<name>.yaml)
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-n', "--name",
                        type=str,
                        default=None,
                        help="Specify name of run (requires: config_<name>.yaml in ./config folder).")
    parser.add_argument('-c', "--config",
                        type=str,
                        help="Specify the full config-file path.")
    parser.add_argument('--nowandb', action='store_true')
    args = parser.parse_args()

    # set wandb to false if nowandb is set
    args.wandb = not args.nowandb

    # read the config file
    config = ReadConfig(args.name, args.config)

    # WandB – Initialize a new run
    if args.wandb:
        wandb.login(key=get_wandb_api_key())
        wandb.init(project="Symmetry-NO",
                   name=config.run_name,
                   group="Darcy",
                   config=config)

    #
    print('Command line inputs: --')
    print('Config name: ', args.name)
    print('Config file: ', args.config, flush=True)

    # run the main training loop
    main(config)
