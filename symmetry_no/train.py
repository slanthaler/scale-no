import sys
import wandb
import argparse

from symmetry_no.data_loader import DarcyData
from symmetry_no.config_helper import ReadConfig
from symmetry_no.wandb_utilities import *
from symmetry_no.fno2d import *
from symmetry_no.selfconsistency import LossSelfconsistency


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
    print('Loading datasets...')
    data = DarcyData(config)

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    modes1 = config.modes
    modes2 = config.modes
    width  = config.width
    depth  = config.depth
    #
    model = FNO2d(modes1,modes2,width,depth).to(device)
    print('FNO2d parameter count: ',count_params(model))

    #
    batch_size = config.batch_size
    epochs = config.epochs
    start_selfcon = config.epochs_selfcon
    track_selfcon = config.track_selfcon
    n_train = config.n_train
    n_test = config.n_test
    use_augmentation = config.augmentation_loss
    iterations = epochs*max(1,n_train//batch_size)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    if args.wandb:
       wandb.watch(model, log_freq=20, log="all")

    loss_fn = LpLoss(size_average=False)
    loss_tracker = {}
    for epoch in range(1, epochs + 1): # config.epochs
        #
        loss_tracker[epoch] = {}
        #
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_aug = 0
        train_sc = 0
        
        # training loop
        for d in data.train_loader:
            x,y = d['train']
            x,y = x.to(device),y.to(device)

            #supervised training
            optimizer.zero_grad()
            out = model(x)
            # DO WE NEED TO NORMALIZE THE OUTPUT??

            loss = loss_fn(out.view(batch_size,-1),y.view(batch_size,-1))
            train_l2 += loss.item()

            # augmentation via sub-sampling
            if use_augmentation:
                loss_aug = LossSelfconsistency(model,x,loss_fn,y)
                loss += 0.5 * loss_aug
                train_aug += loss_aug.item()


            # unsupervised training (selfconsistency constraint)
            if data.selfcon and (track_selfcon or epoch>=start_selfcon):
                x_sc = d['selfcon'][0]
                x_sc = x_sc.to(device)
                #
                # loss_sc = LossSelfconsistency(model,x_sc,loss_fn)
                loss_sc = LossSelfconsistency(model, x_sc, loss_fn)
                if epoch>=start_selfcon:
                    loss += 0.25 * loss_sc

                train_sc += loss_sc.item()

            #
            loss.backward()
            optimizer.step()
            scheduler.step()

        #
        model.eval()
        test_l2 = np.zeros((len(data.test_loaders),))
        with torch.no_grad():
            for i,test_loader in enumerate(data.test_loaders):
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    out = model(x)
                    # DO WE NEED TO NORMALIZE THE OUTPUT??
                    test_l2[i] += loss_fn(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        # normalize losses
        train_l2 /= n_train
        train_sc /= n_train
        train_aug /= n_train
        test_l2 /= n_test

        # track losses
        loss_tracker[epoch]['train_l2'] = train_l2
        loss_tracker[epoch]['train_sc'] = train_sc
        loss_tracker[epoch]['train_aug'] = train_aug
        loss_tracker[epoch]['test_l2'] = test_l2

        t2 = default_timer()
        if args.wandb:
            wandb.log({'time':t2-t1, 'train_l2':train_l2, 'train_selfcon':train_sc})
            wandb.log({f"test_l2/loss-{ii}": loss for ii, loss in enumerate(test_l2)})
        test_losses = " / ".join([f"{val:.5f}" for val in test_l2])
        print(f'[{epoch+1:3}], time: {t2-t1:.3f}, train: {train_l2:.5f}, test: {test_losses}, train_aug: {train_aug:.5f}, train_sc: {train_sc:.5f}', flush=True)
        
#    # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(model.state_dict(), "model.h5")
    np.save("loss_tracker.npy", loss_tracker)
    if args.wandb:
        wandb.save('model.h5')

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
                    help="Specify the full config-file path.")
    parser.add_argument('--nowandb', action='store_true')
    args = parser.parse_args()

    # set wandb to false if nowandb is set
    args.wandb = not args.nowandb

    # read the config file
    config = ReadConfig(args.name,args.config)

    # WandB – Initialize a new run
    if args.wandb:
        wandb.login(key=get_wandb_api_key())
        wandb.init(project="Symmetry-NO", config=config)
    
    #
    print('Command line inputs: --')
    print('Config name: ',args.name)
    print('Config file: ',args.config, flush=True)

    # run the main training loop
    main(config)
