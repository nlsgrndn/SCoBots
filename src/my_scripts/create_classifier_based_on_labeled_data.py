import torch
from eval.classify_z_what import ZWhatEvaluator
from engine.utils import get_config
cfg, task = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nb_used_sample = 32
# z_what_train = torch.randn((400, 32))
# train_labels = torch.randint(high=8, size=(400,))
#z_what_train = torch.load(f"labeled/z_what_validation.pt")
#train_labels = torch.load(f"labeled/labels_validation.pt")
# same but with train
dataset_mode  = "test"
z_what_train = torch.load(f"labeled/pong/z_what_{dataset_mode}.pt")
train_labels = torch.load(f"labeled/pong/labels_{dataset_mode}.pt")
z_what_train = z_what_train[:nb_used_sample]
train_labels = train_labels[:nb_used_sample]
z_what_evaluator = ZWhatEvaluator(cfg)
z_what_evaluator.evaluate_z_what(z_what_train, train_labels)