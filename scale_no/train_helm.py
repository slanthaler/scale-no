import sys
import wandb
import argparse


from scale_no.data_loader import DarcyData, HelmholtzData, NSData
from scale_no.config_helper import ReadConfig
from scale_no.wandb_utilities import *
from scale_no.models.fno2d import *
from scale_no.models.fno2d_doubled import *
from scale_no.models.fno_u import *
from scale_no.models.fno_re import *
from scale_no.models.unet import UNet2d
from scale_no.selfconsistency import LossSelfconsistency
from scale_no.super_sample import sample_helm


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
        out_channel = 1

    elif config.dataset == 'helmholtz':
        print('Loading Helmholtz datasets...')
        data = HelmholtzData(config)
        n_train = config.n_train
        n_test = config.n_test
        in_channel = 9 # complex input output
        out_channel = 2

    elif config.dataset == 'NS':
        print('Loading NS datasets...')
        data = NSData(config)
        n_train = config.n_train * config.T
        n_test = config.n_test * config.T
        in_channel = 7
        out_channel = 1

    else:
        print("config.dataset should be either 'darcy' or 'NS'.")

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    modes1 = config.modes
    modes2 = config.modes
    width = config.width
    depth = config.depth
    mlp = config.mlp
    scale_informed = config.scale_informed
    frequency_pos_emb = config.frequency_pos_emb
    print(f"Width: {width}, Depth: {depth}, Modes: {modes1}, Scale Informed: {scale_informed}, Frequency Pos Emb: {frequency_pos_emb}")


    ### U-shape FNO
    S = config.S
    modes = modes1
    modes_list = []
    width_list = []
    for i in range(10):
        n = 2**i
        modes_list.append(modes//n)
        width_list.append(n*width)

    if config.model == "Unet" or config.model == "UNet":
        model = UNet2d(in_dim=in_channel, out_dim=out_channel, latent_size=S).to(device)
    elif config.model == "FNO":
        model = FNO2d(modes1, modes2, width, depth, in_channel=in_channel, out_channel=out_channel, boundary=True).to(device)
    elif config.model == "FNO_u":
        model = FNO_U(modes_list, modes_list, width_list, level=config.level, depth=depth, mlp=mlp, in_channel=in_channel,
                      out_channel=out_channel, boundary=True, re_log=False).to(device)
    elif config.model == "FNO_re":
        model = FNO_mlp(width, modes1, modes2, depth, scale_informed=scale_informed, frequency_pos_emb=frequency_pos_emb,
                        mlp=mlp, in_channel=in_channel, out_channel=out_channel, sub=0, grid_feature=0, boundary=True, re_log=False).to(device)
    else:
        raise NotImplementedError("model not implement")
    print('FNO2d parameter count: ', count_params(model))

    # model.load_state_dict(torch.load("helm_model_refno.h5"))

    batch_size = config.batch_size
    epochs = config.epochs
    epoch_test = config.epoch_test
    start_selfcon = config.epochs_selfcon
    track_selfcon = config.track_selfcon
    augmentation_samples = config.augmentation_samples
    sample_virtual_instance = config.sample_virtual_instance

    augmentation_loss = config.augmentation_loss
    iterations = epochs * max(1, n_train // batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    if args.wandb:
        wandb.watch(model, log_freq=20, log="all")

    loss_fn = LpLoss(size_average=False)
    for epoch in range(0, epochs + 1):  # config.epochs
        #
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_aug = 0
        train_sc = 0

        # training loop
        for i, train_loader in enumerate(data.train_loaders):
            for d in train_loader:
                x, y, re = d
                x, y, re = x.to(device), y.to(device), re.to(device)

                # supervised training
                optimizer.zero_grad()
                out = model(x, re)
                # DO WE NEED TO NORMALIZE THE OUTPUT??

                loss = loss_fn(out.view(batch_size, -1), y.view(batch_size, -1))
                train_l2 += loss.item()

                # augmentation via sub-sampling
                for j in range(augmentation_samples):
                    if augmentation_loss:
                        loss_aug = LossSelfconsistency(model, x, loss_fn, y=y, re=re, size_min=16, type="helmholtz")
                        loss += augmentation_loss * loss_aug
                        train_aug += loss_aug.item()

                    if sample_virtual_instance and (epoch >= start_selfcon):
                        # rate = 1 + torch.rand(1)
                        rate = 1
                        new_x, rate, new_y = sample_helm(input=x, output=y, rate=rate, keepsize=1)
                        # new_x, new_y = x, y
                        new_re = re * rate
                        loss_sc = LossSelfconsistency(model, new_x, loss_fn, re=new_re, type="helmholtz", align_corner=False)
                        loss += 0.2 * loss_sc
                        train_sc += loss_sc.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

        model.eval()
        test_l2 = np.zeros((len(data.test_loaders),))
        if epoch % epoch_test == 0:
            with torch.no_grad():
                for i, test_loader in enumerate(data.test_loaders):
                    for x, y, re in test_loader:
                        x, y, re = x.to(device), y.to(device), re.to(device)

                        out = model(x, re)
                        # DO WE NEED TO NORMALIZE THE OUTPUT??
                        test_l2[i] += loss_fn(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        # normalize losses
        train_l2 /= (n_train * len(data.train_loaders))
        train_sc /= (n_train * len(data.train_loaders) * augmentation_samples)
        train_aug /= (n_train * len(data.train_loaders) * augmentation_samples)
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
    # if args.wandb:
    torch.save(model.state_dict(), "helm_model_sino_aug.pt")
    print("model saved")
    # wandb.save('../model.h5')



#
if __name__ == '__main__':
    # parse command line arguments
    # (need to specify <name> of run = config_<name>.yaml)
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument('-n', "--name",
                        type=str,
                        help="Specify name of run (requires: config_<name>.yaml in ./config folder).")
    parser.add_argument('-c', "--config",
                        type=str,
                        default='config/helmholtz_ablations/config_helmholtz_scale_freq.yaml',
                        help="Specify the full config-file path.")
    parser.add_argument('--nowandb', action='store_true')
    args = parser.parse_args()

    # set wandb to false if nowandb is set
    # args.wandb = not args.nowandb
    args.wandb = False

    # read the config file
    config = ReadConfig(args.name, args.config)

    # WandB – Initialize a new run
    if args.wandb:
        wandb.login(key=get_wandb_api_key())
        wandb.init(project="Symmetry-NO",
                   name=config.run_name,
                   group="Helmholtz",
                   config=config)

    #
    print('Command line inputs: --')
    print('Config name: ', args.name)
    print('Config file: ', args.config, flush=True)

    # run the main training loop
    main(config)
