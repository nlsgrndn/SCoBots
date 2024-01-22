from model import get_model
from eval.space_eval import SpaceEval
from utils.checkpointer import Checkpointer
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from engine.utils import print_info, load_model

def eval(cfg):
    assert cfg.eval.checkpoint in ['best', 'last']
    assert cfg.eval.metric in ['ap_dot5', 'ap_avg']

    print_info(cfg)

    print('Loading model...')
    model, _, _, _, _ = load_model(cfg, mode="eval")

    print('Loading evaluator...')
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    global_step = 100000
    tb_writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step) # tb refers to tensorboard
    evaluator = SpaceEval(cfg, tb_writer, eval_mode="test")

    model.eval()
    evaluator.eval(model, global_step, cfg)
