from utils import Checkpointer
import os.path as osp
import os
from model.space.space import Space
from model.space.time_consistency import TcSpace
import torch


# as path object
#checkpoint_exp_dir = os.path.join("..", "output", "checkpoints", "final", "pong")
#checkpointer = Checkpointer(checkpoint_exp_dir, max_num=4, load_time_consistency=True, add_flow=True)
#raw_model = TcSpace()
#raw_model = raw_model.to("cpu")
##import ipdb; ipdb.set_trace()
#model = checkpointer.load_last('../output/checkpoints/final/pong/model_000000005.pth', raw_model, None, None, "cpu")
#space_model = model.space
#output = space_model(torch.zeros(1, 3, 128, 128))
#print(output.shape)
#print(output)
def my_load(cfg):
    resume = True
    device = "cpu"
    model = TcSpace()
    checkpointdir = os.path.join("..", "output", "checkpoints", "final")
    exp_name = "pong"
    resume_ckpt = "../output/checkpoints/final/pong/model_000000005.pth"
    #start debug session using pdb
    model = model.to(device)
    model.train() # set model to train mode
    optimizer_fg = None
    optimizer_bg = None
    checkpointer = Checkpointer(osp.join(checkpointdir, exp_name), max_num=4,
                                load_time_consistency=True, add_flow=True)
    start_epoch = 0
    global_step = 0
    if resume: # whether to resume training from a checkpoint
        checkpoint = checkpointer.load_last(resume_ckpt, model, optimizer_fg, optimizer_bg, device)
    
    space_model = model.space
    tensor = torch.ones(1, 3, 128, 128)
    tensor[0, :, 0:4, 0:4] = 0
    loss, log = space_model(tensor)
    z_where, z_pres_prob, z_what, z_depth = log['z_where'], log['z_pres_prob'], log['z_what'], log['z_depth']
    z_where, z_what, z_depth = z_where.squeeze().detach().cpu(), z_what.squeeze().detach().cpu(), z_depth.squeeze().detach().cpu()
    z_pres_prob = z_pres_prob.squeeze().detach().cpu()
    z_pres = z_pres_prob > 0.5
    print(z_pres_prob)
    #start debug session using pdb
    #import ipdb; ipdb.set_trace()
