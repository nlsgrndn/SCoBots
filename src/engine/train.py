from model import get_model
from eval.space_eval import SpaceEval
from dataset import get_dataset, get_dataloader
from solver import get_optimizers
from checkpointer import Checkpointer
from metric_logger import MetricLogger
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from vis import get_vislogger
import time
import torch
from torch.nn.utils import clip_grad_norm_
from rtpt import RTPT
from tqdm import tqdm
import shutil
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model.space.time_consistency import MOCLoss


def print_train_info(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

def print_training_games_info(cfg):
    if len(cfg.gamelist) >= 10:
        print("Training on every game")
        suffix = 'all'
    elif len(cfg.gamelist) == 1:
        suffix = cfg.gamelist[0]
        print(f"Training on {suffix}")
    elif len(cfg.gamelist) == 2:
        suffix = cfg.gamelist[0] + "_" + cfg.gamelist[1]
        print(f"Training on {suffix}")
    else:
        print("Can't train")
        exit(1)
    return suffix # Remark: suffix was not used anywhere in original code, but I added as a return value just in case.

def train(cfg, rtpt_active=True):
    print_train_info(cfg)
    print_training_games_info(cfg)

    if rtpt_active: # RTPT: Remaining Time to Process (library by AIML Lab used to rename your processes giving information on who is launching the process, and the remaining time for it)
        rtpt = RTPT(name_initials='TRo', experiment_name='SPACE-Time', max_iterations=cfg.train.max_epochs)
        rtpt.start()

    # data loading
    dataset = get_dataset(cfg, 'train')
    trainloader = get_dataloader(cfg, 'train')

    # model loading
    model = get_model(cfg)
    model = model.to(cfg.device)
    model.train() # set model to train mode
    optimizer_fg, optimizer_bg = get_optimizers(cfg, model)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,)
    moc_loss_instance = MOCLoss()
    start_epoch = 0
    global_step = 0
    if cfg.resume: # whether to resume training from a checkpoint
        checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)


    # prepare logging of training process
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    if os.path.exists(log_path) and len(log_path) > 15 and cfg.logdir and cfg.exp_name and str(cfg.seed):
        shutil.rmtree(log_path)
    tb_writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step) #tb refers to tensorboard
    vis_logger = get_vislogger(cfg)
    metric_logger = MetricLogger()
    if cfg.train.eval_on:
        valset = get_dataset(cfg, 'val')
        # valloader = get_dataloader(cfg, 'val')
        evaluator = SpaceEval(cfg, tb_writer)
    
    # initialize variables for training loop
    print(f'Start training, Global Step: {global_step}, Start Epoch: {start_epoch} Max: {cfg.train.max_steps}')
    never_evaluated = True
    end_flag = False
    start_log = global_step + 1
    base_global_step = global_step
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if end_flag:
            break

        start = time.perf_counter()
        for (img_stacks, motion, motion_z_pres, motion_z_where) in tqdm(trainloader, desc=f"Epoch {epoch}"):

            end = time.perf_counter()
            data_time = end - start

            # eval on validation set
            if (global_step % cfg.train.eval_every == 0 or never_evaluated) and cfg.train.eval_on:
                never_evaluated = False
                print('Validating...')
                start = time.perf_counter()
                eval_checkpoint = [model, optimizer_fg, optimizer_bg, epoch, global_step]
                results = evaluator.eval(model, valset, valset.bb_path, global_step, cfg.device, cfg)
                checkpointer.save_best("precision_relevant", results["precision_relevant"], eval_checkpoint, min_is_better=False)
                print('Validation takes {:.4f}s.'.format(time.perf_counter() - start))
            
            # main training
            start = time.perf_counter()
            model.train()

            # move to device
            img_stacks, motion, motion_z_pres, motion_z_where = img_stacks.to(cfg.device), motion.to(cfg.device), motion_z_pres.to(cfg.device), motion_z_where.to(cfg.device)
            base_loss, log = model(img_stacks, global_step)
            moc_loss, log = moc_loss_instance.compute_loss(motion, motion_z_pres, motion_z_where, log, global_step)
            loss = base_loss + moc_loss
            loss = loss.mean() # In case of using DataParallel
            optimizer_fg.zero_grad(set_to_none=True)
            optimizer_bg.zero_grad(set_to_none=True)
            end = time.perf_counter()
            batch_time = end - start
            metric_logger.update(data_time=data_time)
            metric_logger.update(batch_time=batch_time)
            metric_logger.update(loss=loss.item())
            loss.backward()
            if cfg.train.clip_norm:
                clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
            optimizer_fg.step()
            optimizer_bg.step()

            # logging
            if global_step%20 == 0: # print in console
                log_state(cfg, epoch, global_step, log, metric_logger) 
            if global_step % cfg.train.print_every == 0 or never_evaluated: # log in tensorboard
                log.update({
                    'loss': metric_logger['loss'].median,
                })
                vis_logger.train_vis(model, tb_writer, log, global_step, 'train', cfg, dataset)

            # checkpointing
            if global_step % cfg.train.save_every == 0: # save checkpoint
                start = time.perf_counter()
                checkpointer.save_last(model, optimizer_fg, optimizer_bg, epoch, global_step)
                print('Saving checkpoint takes {:.4f}s.'.format(time.perf_counter() - start))

            start = time.perf_counter()
            global_step += 1
            if global_step > cfg.train.max_steps:
                end_flag = True

                # final evaluation on validation set
                if cfg.train.eval_on:
                    print('Final evaluation on validation set...')
                    start = time.perf_counter()
                    evaluator.eval(model, valset, valset.bb_path, global_step,
                                              cfg.device, cfg)
                    print('Validation takes {:.4f}s.'.format(time.perf_counter() - start))
                
                break

        if rtpt_active:
            rtpt.step()

def log_state(cfg, epoch, global_step, log, metric_logger):
    print()
    print(
        'exp: {}, epoch: {}, global_step: {}, loss: {:.2f}, z_what_con: {:.2f},'
        ' z_pres_con: {:.3f}, z_what_loss_pool: {:.3f}, z_what_loss_objects: {:.3f}, motion_loss: {:.3f}, '
        'motion_loss_z_where: {:.3f} motion_loss_alpha: {:.3f} batch time: '
        '{:.4f}s, data time: {:.4f}s'.format(
            cfg.exp_name, epoch + 1, global_step, metric_logger['loss'].median,
            torch.sum(log['z_what_loss']).item(), torch.sum(log['z_pres_loss']).item(),
            torch.sum(log['z_what_loss_pool']).item(),
            torch.sum(log['z_what_loss_objects']).item(),
            torch.sum(log['flow_loss']).item(),
            torch.sum(log['flow_loss_z_where']).item(),
            torch.sum(log['flow_loss_alpha_map']).item(),
            metric_logger['batch_time'].avg, metric_logger['data_time'].avg))
    print()
