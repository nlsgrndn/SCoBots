from yacs.config import CfgNode
classifier_cfg = CfgNode({
    'train_folder': "val", # classifier should be trained on val sets (train set not used because it contains the objects for which the encodings were optimized -> biased dataset)
    'test_folder': "test",
    'clf_name': "nn",
    'data_subset_mode': "relevant",
    "one_image_per_sequence": True, # use case: avoid many similar images in the dataset
    "visualize": True,
})