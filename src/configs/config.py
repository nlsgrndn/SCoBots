from yacs.config import CfgNode

# dirname = os.checkpointdir.basename(os.checkpointdir.dirname(os.checkpointdir.abspath(__file__)))

cfg = CfgNode({
    'seed': 8848,
    'exp_name': '',
    'model': 'SPACE',
    'flow': False,
    # Resume training or not
    'resume': True,
    # If resume is true, then we load this checkpoint. If '', we load the last checkpoint
    'resume_ckpt': '',
    # Whether to use multiple GPUs
    'parallel': False,
    # Device ids to use
    'device_ids': [0, 1],
    'device': 'cuda:0',
    'logdir': '../output/logs/',
    'checkpointdir': '../output/checkpoints/',
    'evaldir': '../output/eval/',
    'demodir': '../output/demo/',
    'save_relevant_objects': False,
    # Dataset to use
    'dataset': 'OBJ3D_LARGE',
    'dataset_style': 'space_like',

    'dataset_roots': {
        'ATARI': '../data/ATARI',
        'OBJ3D_LARGE': '../data/OBJ3D_LARGE',
        'OBJ3D_SMALL': '../data/OBJ3D_SMALL',
    },

    # For Atari
    'gamelist': [
        'Atlantis-v0',
        'Asterix-v0',
        'Carnival-v0',
        'DoubleDunk-v0',
        'Kangaroo-v0',
        'MontezumaRevenge-v0',
        'MsPacman-v0',
        'Pooyan-v0',
        'Qbert-v0',
        'SpaceInvaders-v0',
        'Pong-v0',
        'Tennis-v0',
    ],


    # For engine.train
    'train': {
        'log': True,
        'batch_size': 16,
        'max_epochs': 1000,
        'max_steps': 1000000,

        'num_workers': 4,
        # Gradient clipping. If 0.0 we don't clip
        'clip_norm': 1.0,
        'max_ckpt': 5,

        'print_every': 500,
        'save_every': 1000,
        'eval_on': True,
        'eval_every': 1000,

        'solver': {
            'fg': {
                'optim': 'RMSprop',
                'lr': 1e-5,
            },
            'bg': {
                'optim': 'Adam',
                'lr': 1e-3,
            }
        },
        'black_background': False,  
        'dilation': False,
    },

    # For engine.eval
    'eval': {
        # One of 'best', 'last'
        'checkpoint': 'best',
        # Either 'ap_dot5' or 'ap_avg'
        'metric': 'ap_avg'
    },

    # For engine.show_images
    'show': {
        # Either 'val' or 'test'
        'mode': 'val',
        # Indices into the dataset
        'indices': [0]
    }
})

from  space_models.space.arch import arch
from configs.eval_cfg import eval_cfg
from configs.dataset_size_cfg import dataset_size_cfg
from configs.classifier_cfg import classifier_cfg

# For these three, please go to the corresponding file
cfg.arch = arch
cfg.eval_cfg = eval_cfg
cfg.dataset_size_cfg = dataset_size_cfg
cfg.classifier = classifier_cfg