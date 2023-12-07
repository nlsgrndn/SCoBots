from niceprint import makedirs, pprint as print


import torch
from torch import nn


import json
import math
import os
import os.path as osp
import pickle
from datetime import datetime


class Checkpointer:
    def __init__(self, checkpointdir, max_num,):
        self.max_num = max_num
        self.checkpointdir = checkpointdir
        if not osp.exists(checkpointdir):
            makedirs(checkpointdir)
        self.listfile = osp.join(checkpointdir, 'model_list.pkl')
        print(f'Listfile: {self.listfile}')
        if not osp.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)

    def save(self, path: str, model, optimizer_fg, optimizer_bg, epoch, global_step):
        assert path.endswith('.pth')
        makedirs(osp.dirname(path), exist_ok=True)

        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer_fg': optimizer_fg.state_dict() if optimizer_fg else None,
            'optimizer_bg': optimizer_bg.state_dict() if optimizer_bg else None,
            'epoch': epoch,
            'global_step': global_step
        }
        with open(path, 'wb') as f:
            torch.save(checkpoint, f)
            print('blue', f'Checkpoint has been saved to "{path}".')

    def save_last(self, model, optimizer_fg, optimizer_bg, epoch, global_step):
        path = osp.join(self.checkpointdir, 'model_{:09}.pth'.format(global_step + 1))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if osp.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(path)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
        self.save(path, model, optimizer_fg, optimizer_bg, epoch, global_step)

    def load(self, path, model, optimizer_fg=None, optimizer_bg=None, device=None):
        """
        Return starting epoch and global step
        """

        assert osp.exists(path), f'Checkpoint {path} does not exist.'
        checkpoint = torch.load(path, map_location=device)
        checkpoint_model = checkpoint.pop('model')
        try:
            model.load_state_dict(checkpoint_model)
        except RuntimeError as rterr:
            model.space.load_state_dict(checkpoint_model)
        if optimizer_fg:
            optimizer_fg.load_state_dict(checkpoint.pop('optimizer_fg'))
        if optimizer_bg:
            optimizer_bg.load_state_dict(checkpoint.pop('optimizer_bg'))
        print('blue', f'Checkpoint "{path}" loaded.')
        return checkpoint

    def load_last(self, path, model, optimizer_fg=None, optimizer_bg=None, device=None):
        """
        If path is '', we load the last checkpoint
        """
        if path == '':
            with open(self.listfile, 'rb') as f:
                model_list = pickle.load(f)
                if len(model_list) == 0:
                    print('yellow', 'No checkpoint found. Starting from scratch')
                    return None
                else:
                    path = model_list[-1]
        if not path.startswith('../'):
            path = '../' + path
        return self.load(path, model, optimizer_fg, optimizer_bg, device)

    def save_best(self, metric_name, value, checkpoint, min_is_better):
        metric_file = os.path.join(self.checkpointdir, f'best_{metric_name}.json')
        checkpoint_file = os.path.join(self.checkpointdir, f'best_{metric_name}.pth')

        now = datetime.now()
        log = {
            'name': metric_name,
            'value': float(value),
            'date': now.strftime("%Y-%m-%d %H:%M:%S"),
            'global_step': checkpoint[-1]
        }

        if not os.path.exists(metric_file):
            dump = True
        else:
            with open(metric_file, 'r') as f:
                previous_best = json.load(f)
            if not math.isfinite(log['value']):
                dump = True
            elif (min_is_better and log['value'] < previous_best['value']) or (
                    not min_is_better and log['value'] > previous_best['value']):
                dump = True
            else:
                dump = False
        if dump:
            with open(metric_file, 'w') as f:
                json.dump(log, f)
            self.save(checkpoint_file, *checkpoint)

    def load_best(self, metric_name, model, fg_optimizer, bg_optimizer, device):
        metric_file = os.path.join(self.checkpointdir, f'best_{metric_name}.json')
        checkpoint_file = os.path.join(self.checkpointdir, f'best_{metric_name}.pth')

        assert osp.exists(metric_file), 'Metric file does not exist'
        assert osp.exists(checkpoint_file), 'checkpoint file does not exist'

        return self.load(checkpoint_file, model, fg_optimizer, bg_optimizer, device)