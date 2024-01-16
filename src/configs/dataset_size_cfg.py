from yacs.config import CfgNode
dataset_size_cfg = CfgNode({
    'max_num_of_different_samples_for_dataset_mode':{
        "train": 2048,
        "val": 128,
        "test": 128}
})