from yacs.config import CfgNode
eval_cfg = CfgNode({
    # Evaluation during training
    'train': {
        # What to evaluate
        'metrics': ['ap', 'mse',], #['ap', 'mse', 'cluster'],
        # Number of samples for evaluation
        'num_samples': {
            'mse': 1024,
            'ap': 1024,
            'cluster': 1024,
        },

        # For dataloader
        'batch_size': 1, #32,
        'num_workers': 1, #4,
    },
    'test': {
        # For dataloader
        'batch_size': 1, #2,
        'num_workers': 1, #4,
    }

})
