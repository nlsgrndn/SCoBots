from checkpointer import Checkpointer
import os.path as osp
import os
from model.space.space import Space
from model.space.time_consistency import TcSpace
import torch


def my_load(cfg):
    resume = True
    device = "cpu"
    model = TcSpace()
    checkpointdir = os.path.join("..", "output", "checkpoints", "final")
    exp_name = "pong"
    resume_ckpt = "../old_models/model_000005001.pth"#"../output/checkpoints/final/pong/model_000000029.pth"
    #start debug session using pdb
    model = model.to(device)
    model.train() # set model to train mode
    optimizer_fg = None
    optimizer_bg = None
    checkpointer = Checkpointer(osp.join(checkpointdir, exp_name), max_num=4,)
    start_epoch = 0
    global_step = 0
    if resume: # whether to resume training from a checkpoint
        checkpoint = checkpointer.load_last(resume_ckpt, model, optimizer_fg, optimizer_bg, device)
    
    space_model = model.space
    # load image using matplotlib
    image_path = "engine/00000_0.png"
    import matplotlib.pyplot as plt
    import numpy as np
    image = plt.imread(image_path)
    image = torch.from_numpy(image)
    # reshape from (128, 128, 3) to (1, 3, 128, 128)
    image = image.permute(2, 0, 1)
    #import ipdb; ipdb.set_trace()
    #tensor = torch.ones(1, 3, 128, 128)
    #tensor[0, :, 0:4, 0:4] = 0
    loss, log = space_model(image.unsqueeze(0))
    z_where, z_pres_prob, z_what, z_depth = log['z_where'], log['z_pres_prob'], log['z_what'], log['z_depth']
    z_where, z_what, z_depth = z_where.squeeze().detach().cpu(), z_what.squeeze().detach().cpu(), z_depth.squeeze().detach().cpu()
    z_pres_prob = z_pres_prob.squeeze().detach().cpu()
    z_pres = z_pres_prob > 0.5
    print(z_pres.shape)
    # transfrom z_pres_prob to 16x16
    z_pres_prob = z_pres_prob.view(16, 16)
    print(z_pres_prob)
    plt.imsave("engine/z_pres_prob.png", z_pres_prob)
    #start debug session using pdb
    #import ipdb; ipdb.set_trace()
