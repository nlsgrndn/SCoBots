from yacs.config import CfgNode
eval_cfg = CfgNode({
    'val': {
        # What to evaluate
        'metrics': ['ap','cluster'], #['ap', 'mse', 'cluster'],
        # Number of samples for evaluation #TODO currently not used
        'num_samples': {
            'mse': 1024,
            'ap': 1024,
            'cluster': 1024,
        },

        # For dataloader
        'batch_size': 16, 
        'num_workers': 8,
    },
    'test': {
        # What to evaluate
        'metrics': ['ap','cluster'], #['ap', 'mse', 'cluster'],
        # Number of samples for evaluation #TODO currently not used
        'num_samples': {
            'mse': 1024,
            'ap': 1024,
            'cluster': 1024,
        },

        # For dataloader
        'batch_size': 16, 
        'num_workers': 8,
    },
})
