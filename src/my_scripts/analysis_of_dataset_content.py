import os
import torch
# aiml_atari_data/Pong-v0/mode/test/00000_0_128.pt
path_128 = os.path.join("..", "aiml_atari_data", "Pong-v0", "flow", "test", "00000_0_128.pt")
path_z_where = os.path.join("..","aiml_atari_data", "Pong-v0", "flow", "test", "00000_0_z_where.pt")
path_z_pres = os.path.join("..","aiml_atari_data", "Pong-v0", "flow", "test", "00000_0_z_pres.pt")
# read the files
tensor_128 = torch.load(path_128)
tensor_z_where = torch.load(path_z_where)
tensor_z_pres = torch.load(path_z_pres)
# check the shape of the tensors
print(tensor_128.shape)
print(tensor_z_where.shape)
print(tensor_z_pres.shape)
print(tensor_128)
print(tensor_128.max(), tensor_128.min())
#check whether only zero values are present
#print(torch.all(tensor == 0))
#print indices of non-zero values
#print(torch.nonzero(tensor))
#tranform tensor to numpy array
npy_128 = tensor_128.numpy()
# save numpy array as image
import matplotlib.pyplot as plt
# create folder flow_visualization
# check whether folder exists
if not os.path.exists("flow_visualization"):
    os.mkdir("flow_visualization")
for i in range(0,9):
    path_128 = os.path.join("..", "aiml_atari_data", "Pong-v0", "flow", "train", f"0000{i}_0_128.pt")
    tensor_128 = torch.load(path_128)
    npy_128 = tensor_128.numpy()
    plt.imsave(f"flow_visualization/test_{i}.png", npy_128, cmap="gray")
print (tensor_z_pres)
print (tensor_z_where)
