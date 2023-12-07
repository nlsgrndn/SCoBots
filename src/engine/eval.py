from model import get_model
from eval import get_evaluator #comes from __init__.py
from dataset import get_dataset, get_dataloader
from checkpointer import Checkpointer
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def eval(cfg):
    assert cfg.eval.checkpoint in ['best', 'last']
    assert cfg.eval.metric in ['ap_dot5', 'ap_avg']
    #if True:
    #    cfg.device = 'cuda:0'
    #    cfg.device_ids = [0]

    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'see below')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

    print('Loading data')
    testset = get_dataset(cfg, 'test')
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    evaluator = get_evaluator(cfg)
    model.eval()

    if cfg.resume_ckpt:
        checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)
    elif cfg.eval.checkpoint == 'last':
        checkpointer.load_last('', model, None, None, cfg.device)
    elif cfg.eval.checkpoint == 'best':
        checkpointer.load_best(cfg.eval.metric, model, None, None, cfg.device)
    if cfg.parallel:
        assert 'cpu' not in cfg.device
        model = nn.DataParallel(model, device_ids=cfg.device_ids)

    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    global_step = 100000
    writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step)
    evaluator.eval(model, testset, testset.bb_path, writer, global_step,
                        cfg.device, cfg)
