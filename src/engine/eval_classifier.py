import pandas as pd
from dataset.atari_data_collector import AtariDataCollector
from engine.utils import load_classifier
from sklearn.metrics import classification_report
# import create_latent_dataset
from create_latent_dataset import create_latent_dataset

def eval_classifier(cfg):
    dataset_mode = cfg.classifier.test_folder
    data_subset_mode = cfg.classifier.data_subset_mode
    clf_name = cfg.classifier.clf_name
    only_collect_first_image_of_consecutive_frames = cfg.classifier.one_image_per_sequence

    folder_path = cfg.resume_ckpt.rsplit('/', 1)[0]

    folder_path = folder_path.replace("space_weights", "classifier")

    clf, centroid_labels = load_classifier(
        folder_path=folder_path,
        clf_name=clf_name,
        data_subset_mode=data_subset_mode
    )
    
    # create and collect the data
    create_latent_dataset(cfg, dataset_mode)
    x, y = AtariDataCollector.collect_z_what_data_reshaped(cfg, dataset_mode, data_subset_mode, only_collect_first_image_of_consecutive_frames)
    x, y = x.cpu(), y.cpu() # put all data onto cpu

    print("Evaluating classifier on", len(x), "z_what encodings")

    pred_y = clf.predict(x)
    if centroid_labels is not None:
        pred_y = [centroid_labels[pred_y_i] for pred_y_i in pred_y]
 
    metric_dict = classification_report(y, pred_y, output_dict=True)
    for k, v in metric_dict.items():
        print(f"{k}: {v}")
    
    # save metric_dict as csv
    df = pd.DataFrame(metric_dict)
    df.to_csv(f"{folder_path}/eval_classifier.csv", header=True, index=True)

