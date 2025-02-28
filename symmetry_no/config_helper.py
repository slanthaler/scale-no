from ruamel.yaml import YAML
from argparse import Namespace 

from symmetry_no.rootdir import ROOT_DIR

# taken from [ https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8 ]
class RecursiveNamespace(Namespace):

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
    
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    

def DefaultConfig():
    #
    config = dict(
        name='default',
        seed=0,
        # fno2d
        modes=12,
        width=16,
        # depth=4,
        # training
        epochs=200,
        epochs_selfcon=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        use_augmentation=dict(
            CropResize={'p': 0.5, 
                        'scale_min': 0.1,
                        'size_min': 32},
            Flip={'p': 0.5}
        ),
        grid_size=None,
        n_train=16,
        n_test=8,
        # datasets
        train_data='',
        test_data='',
        selfcon_data='',
        # track selfconsistency loss?
        track_selfcon=True,
    )
    return config

def CleanUpNone(config):
    for key,val in config.items():
        if isinstance(val,dict):
            config[key] = CleanUpNone(val)
        if isinstance(val,str):
            if val.lower()=='none':
                config[key] = None

    return config

def ReadConfig(name,config_file):
    if name:           
        if config_file:
            ValueError('--name and --config flags are mutually exclusive!')
        #
        config_file = ROOT_DIR + '/config/darcy/config_' + name + '.yaml'
    
    yaml = YAML()
    with open(config_file, "r") as f:
        config_user = yaml.load(f)
    # 
    config = DefaultConfig()
    for key in config_user:
        config[key] = config_user[key]

    # make sure None is read as type None, not <'None', type(str)>
    config = CleanUpNone(config)

    # prefer namespace over dictionary
    config = RecursiveNamespace(**config)

    return config
