from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

def DefaultConfig():
    #
    config = dict(
        name='default',
        seed=42,
        # fno2d
        modes=12,
        width=16,
        depth=4,
        # training
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=32,
        use_augmentation=dict(
            CropResize={'p': 0.5, 
                        'scale_min': 0.1,
                        'size_min': 32},
            Flip={'p': 0.5}
        ),
        grid_size=128,
        n_train=16,
        n_test=8,
        n_selfcon=0,
        # datasets
        train_data='',
        test_data='',
        selfcon_data='',
    )
    return config

def ReadConfig(run_name):
    # read-in user supplied key values
    config_file = 'config_' + run_name + '.yaml'
    pipe = ConfigPipeline([
        YamlConfig(config_file, config_folder="./config"),
    ])
    config_user = pipe.read_conf()
    
    # 
    config = DefaultConfig()
    for key in config_user:
        config[key] = config_user[key]

    return config
