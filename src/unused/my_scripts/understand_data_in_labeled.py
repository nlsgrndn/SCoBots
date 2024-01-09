import torch

dataset_mode = "train"
z_what_train = torch.load(f"labeled/z_what_{dataset_mode}.pt")
train_labels = torch.load(f"labeled/labels_{dataset_mode}.pt")

print(z_what_train.shape)
print(train_labels.shape)
print(z_what_train[0])