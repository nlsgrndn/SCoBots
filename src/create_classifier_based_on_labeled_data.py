import torch
from eval.classify_z_what import evaluate_z_what
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
evaluate_z_what({'indices':None, 'dim': 2, 'edgecolors': True}, z_what_train, train_labels, nb_used_sample, cfg=cfg)