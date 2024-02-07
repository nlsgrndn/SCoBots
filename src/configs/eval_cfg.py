from yacs.config import CfgNode
eval_cfg = CfgNode({
    'val': {
        # What to evaluate
        'metrics': ['ap','cluster'], #['ap', 'mse', 'cluster'],

        # For dataloader
        'batch_size': 16, 
        'num_workers': 8,
    },
    'test': {
        # What to evaluate
        'metrics': ['cluster', 'ap'], #['ap', 'mse', 'cluster'],

        # For dataloader
        'batch_size': 16, 
        'num_workers': 8,
    },
})
