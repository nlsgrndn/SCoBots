import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg
from .space import Space
from rtpt import RTPT
import time

class MOCLoss():
    def __init__(self):
        self.zero = nn.Parameter(torch.tensor(0.0))
        self.area_object_weight = arch.area_object_weight
        self.count_difference_variance = RunningVariance()
  
    def compute_loss(self, motion, motion_z_pres, motion_z_where, logs, global_step):
        """
        Inference.
        With time-dimension for consistency
        :param x: (B, 3, H, W)
        :param motion: (B, H, W)
        :param motion_z_pres: z_pres hint from motion
        :param motion_z_where: z_where hint from motion
        :param global_step: global training step
        :return:
            loss: a scalar. Note it will be better to return (B,)
            log: a dictionary for visualization
        """
        if len(motion.shape) == 4: # (B, T, H, W)
            B, T, H, W = motion.shape
            motion = motion.reshape(T * B, 1, H, W)
            motion_z_pres = motion_z_pres.reshape(T * B, arch.G * arch.G, 1)
            motion_z_where = motion_z_where.reshape(T * B, arch.G * arch.G, 4)
        else: # (B, H, W)
            B, H, W = motion.shape
            motion = motion.reshape(B, 1, H, W)
            motion_z_pres = motion_z_pres.reshape(B, arch.G * arch.G, 1)
            motion_z_where = motion_z_where.reshape(B, arch.G * arch.G, 4)

        tc_log = {
            'motion': motion,
            'motion_z_pres': motion_z_pres,
            'motion_z_where': motion_z_where,
        }
        logs.update(tc_log)

        z_what_loss = z_what_loss_grid_cell(logs) if arch.adjacent_consistency_weight > 1e-3 else self.zero
        z_pres_loss = z_pres_loss_grid_cell(logs) if arch.pres_inconsistency_weight > 1e-3 else self.zero
        z_what_loss_pool = z_what_consistency_pool(logs) if arch.area_pool_weight > 1e-3 else self.zero
        z_what_loss_objects, objects_detected = z_what_consistency_objects(logs) if arch.area_object_weight > 1e-3 else (self.zero, self.zero)
        flow_loss = compute_flow_loss(logs) if arch.motion_loss_weight_z_pres > 1e-3 else self.zero
        flow_loss_z_where = compute_flow_loss_z_where(logs) if arch.motion_loss_weight_z_where > 1e-3 else self.zero
        flow_loss_alpha_map = compute_flow_loss_alpha(logs, motion) if arch.motion_loss_weight_alpha > 1e-3 else self.zero

        if arch.dynamic_scheduling:
            object_count_accurate = self.object_count_accurate_scaling(logs)
            area_object_scaling = arch.dynamic_steepness ** (-object_count_accurate)
            flow_scaling = 1 - area_object_scaling
        else:
            flow_scaling = max(0, 1 - (global_step - arch.motion_cooling_start_step) / arch.motion_cooling_end_step) # weird division by 2 removed
            area_object_scaling = 1 - flow_scaling
            flow_scaling = torch.tensor(flow_scaling).to(motion.device)
            area_object_scaling = torch.tensor(area_object_scaling).to(motion.device)

        motion_loss = (flow_loss * arch.motion_loss_weight_z_pres * flow_scaling \
            + flow_loss_alpha_map * arch.motion_loss_weight_alpha * flow_scaling \
            + flow_loss_z_where * arch.motion_loss_weight_z_where * flow_scaling) * arch.motion_weight
        
        # with standard param settings only arch.area_object_weight != 0.0 and only iff MOC is used
        combined_z_what_loss = z_what_loss * arch.adjacent_consistency_weight \
            + z_pres_loss * arch.pres_inconsistency_weight \
            + z_what_loss_pool * arch.area_pool_weight \
            + z_what_loss_objects * area_object_scaling * arch.area_object_weight
        
        moc_loss = motion_loss + combined_z_what_loss

        tc_log = {
            'z_what_loss': z_what_loss,
            'z_what_loss_pool': z_what_loss_pool,
            'z_what_loss_objects': z_what_loss_objects,
            'z_pres_loss': z_pres_loss,
            'flow_loss': flow_loss,
            'flow_loss_z_where': flow_loss_z_where,
            'flow_loss_alpha_map': flow_loss_alpha_map,
            'objects_detected': objects_detected,
            'flow_scaling': flow_scaling,
            'area_object_scaling': area_object_scaling,
            'total_loss': moc_loss, # TODO rename to moc_loss, and change in space_vis for logging
            'motion_loss': motion_loss,
            'motion_loss_no_flow_scaling': motion_loss / flow_scaling,
            'combined_z_what_loss': combined_z_what_loss,
            'scaled_flow_loss': flow_loss * arch.motion_loss_weight_z_pres * flow_scaling * arch.motion_weight,
            'scaled_flow_loss_z_where': flow_loss_z_where * arch.motion_loss_weight_z_where * flow_scaling * arch.motion_weight,
            'scaled_flow_loss_alpha_map': flow_loss_alpha_map * arch.motion_loss_weight_alpha * flow_scaling * arch.motion_weight,
            'scaled_z_what_loss': z_what_loss * arch.adjacent_consistency_weight,
            'scaled_z_pres_loss': z_pres_loss * arch.pres_inconsistency_weight,
            'scaled_z_what_loss_pool': z_what_loss_pool * arch.area_pool_weight,
            'scaled_z_what_loss_objects': z_what_loss_objects * area_object_scaling * arch.area_object_weight,
        }
        logs.update(tc_log)
        return moc_loss, logs
    
    def object_count_accurate_scaling(self, responses):
        # (T, B, G, G, 5)
        z_whats = responses['z_what']
        _, GG, D = z_whats.shape
        # (B, T, G*G, 1)
        z_whats = z_whats.reshape(-1, arch.T, GG, D)
        B, T = z_whats.shape[:2]
        z_where_pure = responses['z_where_pure'].reshape(B, T, arch.G, arch.G, -1).transpose(0, 1)
        z_pres_pure = responses['z_pres_prob_pure'].reshape(B, T, GG, 1).reshape(B, T, arch.G, arch.G).transpose(0, 1)
        motion_z_where = responses['motion_z_where'].reshape(B, T, arch.G, arch.G, -1).transpose(0, 1)
        motion_z_pres = responses['motion_z_pres'].reshape(B, T, GG, 1).reshape(B, T, arch.G, arch.G).transpose(0, 1)
        frame_marker = (torch.arange(T * B, device=z_pres_pure.device) * 10).reshape(T, B)[..., None, None].expand(T, B, arch.G, arch.G)[..., None]
        z_where_pad = torch.cat([z_where_pure, frame_marker], dim=4)
        motion_pad = torch.cat([motion_z_where, frame_marker], dim=4)
        # (#hits, 5)
        objects = z_where_pad[z_pres_pure > arch.object_threshold]
        motion_objects = motion_pad[motion_z_pres > arch.object_threshold]
        if objects.nelement() == 0 or motion_objects.nelement() == 0:
            return 1000

        # (#objects, #motion_objects)
        dist = torch.cdist(motion_objects, objects, p=1)
        # (#hits, 1)
        values, _ = torch.topk(dist, 1, largest=False)
        motion_z_pres = responses['motion_z_pres'].mean().detach().item()
        pred_z_pres = responses['z_pres_prob_pure'].mean().detach().item()
        zero = 0

        motion_objects_found = values.mean().detach().item()
        # print(f'{motion_objects_found=}')
        value = (max(zero, (motion_objects_found - arch.z_where_offset) * arch.motion_object_found_lambda) +
                 max(zero, pred_z_pres - motion_z_pres * arch.motion_underestimating))
        self.count_difference_variance += pred_z_pres - motion_z_pres
        return (value + self.count_difference_variance.value() * arch.use_variance)\
               * responses['motion_z_pres'][0].nelement()
    
class RunningVariance:
    def __init__(self, n=arch.variance_steps):
        self.values = []
        self.n = n

    def __add__(self, other):
        if not self.values:
            self.values.append(other)
        self.values.append(other)
        self.values = self.values[-self.n:]
        return self

    def value(self):
        return torch.var(torch.tensor(self.values), unbiased=True)


def sq_delta(t1, t2):
    return (t1 - t2).square()


def get_log(res):
    return res[1]


def get_loss(res):
    loss = res[0]
    return loss


def compute_flow_loss(responses):
    z_pres = responses['z_pres_prob_pure']
    motion_z_pres = responses['motion_z_pres']
    return nn.functional.mse_loss(z_pres, motion_z_pres.reshape(z_pres.shape), reduction='sum')


def compute_flow_loss_alpha(responses, motion):
    alpha_map = responses['alpha_map']
    alpha_map_gt = motion
    return nn.functional.mse_loss(alpha_map, alpha_map_gt, reduction='sum')


# Variants:
# 1. Slow: Maximize IOU over preds and motion bbs
# 2. Fast Approx: z_pres is lower or much higher than
# 3. Fast: Use both way mse
# 4. Medium: 2. + MSE of close by bb
# 5. Unknown: MSE of Spatial Transform Masks of pred and motion bbs, where more pred is not so bad

# 3. Assumes SPACE takes same z_pres like motion oracle
def compute_flow_loss_z_where(responses):
    motion_z_pres = responses['motion_z_pres'].squeeze()
    pred_z_pres = responses['z_pres_prob_pure'].reshape(motion_z_pres.shape)
    z_where = responses['z_where_pure']
    motion_z_where = responses['motion_z_where']
    return nn.functional.mse_loss(z_where[motion_z_pres > 0.5], motion_z_where[motion_z_pres > 0.5], reduction='sum') + \
           nn.functional.mse_loss(z_where[pred_z_pres > 0.5], motion_z_where[pred_z_pres > 0.5], reduction='sum')


# Align over many pairs of objects -> Push apart only if different color, shape/size
def z_what_consistency_objects(responses):
    cos = nn.CosineSimilarity(dim=1)
    z_whats = responses['z_what']
    _, GG, D = z_whats.shape
    # (B, T, G*G, 1)
    z_whats = z_whats.reshape(-1, arch.T, GG, D)
    B, T = z_whats.shape[:2]

    # (T, B, G, G)
    z_pres = responses['z_pres_prob'].reshape(B, T, GG, 1).reshape(B, T, arch.G, arch.G).transpose(0, 1)
    z_pres_idx = (z_pres[:-1] > arch.object_threshold).nonzero(as_tuple=False)
    # (T, B, G, G, D)
    z_whats = z_whats.reshape(B, T, arch.G, arch.G, D).transpose(0, 1)
    # (T, B, G, G, 4)
    z_where = responses['z_where'].reshape(B, T, arch.G, arch.G, -1).transpose(0, 1)
    # (T, B, G+2, G+2)
    z_pres_same_padding = torch.nn.functional.pad(z_pres, (1, 1, 1, 1), mode='circular')
    # (T, B, G+2, G+2, D)
    z_what_same_padding = torch.nn.functional.pad(z_whats, (0, 0, 1, 1, 1, 1), mode='circular')
    # (T, B, G+2, G+2, 4)
    z_where_same_padding = torch.nn.functional.pad(z_where, (0, 0, 1, 1, 1, 1), mode='circular')
    # idx: (4,)
    object_consistency_loss = torch.tensor(0.0).to(z_whats.device)
    for idx in z_pres_idx:
        # (3, 3)
        z_pres_area = z_pres_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # (3, 3, D)
        z_what_area = z_what_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # (3, 3, 4)
        z_where_area = z_where_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # Tuple((#hits,) (#hits,))
        z_what_idx = (z_pres_area > arch.object_threshold).nonzero(as_tuple=True)
        # (1, D)
        z_what_prior = z_whats[idx.tensor_split(4)]
        # (1, 4)
        z_where_prior = z_where[idx.tensor_split(4)]
        # (#hits, D)
        z_whats_now = z_what_area[z_what_idx]
        # (#hits, 4)
        z_where_now = z_where_area[z_what_idx]
        if z_whats_now.nelement() == 0:
            continue
        # (#hits,)
        if arch.cosine_sim:
            z_sim = cos(z_what_prior, z_whats_now)
        else:
            z_means = nn.functional.mse_loss(z_what_prior, z_whats_now, reduction='none')
            # (#hits,) in (0,1]
            z_sim = 1 / (torch.mean(z_means, dim=1) + 1)
        if z_whats_now.shape[0] > 1:
            similarity_max_idx = z_sim.argmax()
            if arch.agree_sim:  # True
                pos_dif_min = nn.functional.mse_loss(z_where_prior.expand_as(z_where_now), z_where_now,
                                                     reduction='none').sum(dim=-1).argmin()
                # color_dif_min = color_match(
                #     responses['imgs'][idx[1] * B + idx[0]],
                #     responses['imgs'][idx[1] * B + idx[0] + 1],
                #     z_where_prior,
                #     z_where_now
                # )
                if pos_dif_min != similarity_max_idx:
                    continue
            object_consistency_loss += -arch.z_cos_match_weight * z_sim[similarity_max_idx] + torch.sum(z_sim)
        else:
            object_consistency_loss += -arch.z_cos_match_weight * torch.max(z_sim) + torch.sum(z_sim)
    return object_consistency_loss, torch.tensor(len(z_pres_idx)).to(z_whats.device)



def z_what_consistency_pool(responses):
    # O(n^2) matching of only high z_pres
    # (T, B, G*G, D)
    z_whats = responses['z_what']
    _, GG, D = z_whats.shape
    z_whats = torch.abs(z_whats).reshape(arch.T, -1, GG, D)
    T, B = z_whats.shape[:2]
    # (T, B, G*G, 1)
    z_pres = responses['z_pres_prob'].reshape(T, -1, GG, 1)

    z_what_weighted = z_whats * z_pres
    pool = nn.MaxPool2d(3, stride=1, padding=1)
    # (T * B, D, G, G)
    z_what_reshaped = z_what_weighted.transpose(2, 3).reshape(T, B, D, arch.G, arch.G).reshape(T * B, D, arch.G, arch.G)
    z_what_smoothed = pool(z_what_reshaped)
    # (T, B, G * G, D)
    z_what_smoothed = z_what_smoothed.reshape(T, B, D, arch.G, arch.G).reshape(T, B, D, arch.G * arch.G).transpose(2, 3)
    if arch.cosine_sim:
        # (T-1, B, G * G)
        cos = nn.CosineSimilarity(dim=3, eps=1e-6)(z_what_smoothed[:-1], z_what_weighted[1:])
        z_pres_weight = z_pres.squeeze()
        z_pres_weight = (z_pres_weight[:-1] + z_pres_weight[1:]) * 0.5
        return -torch.sum(cos * z_pres_weight)
    else:
        return torch.sum(nn.functional.mse_loss(z_what_smoothed[:-1], z_what_weighted[1:], reduction='none'))


def z_what_loss_grid_cell(responses):
    # (T, B, G*G, D)
    _, GG, D = responses['z_what'].shape
    z_whats = responses['z_what'].reshape(arch.T, -1, GG, D)
    # (T-1, B, G*G, D)
    z_what_deltas = sq_delta(z_whats[1:], z_whats[:-1])
    return torch.sum(z_what_deltas)


def z_pres_loss_grid_cell(responses):
    z_press = responses['z_pres_prob'].reshape(arch.T, -1, arch.G * arch.G, 1)
    # (T-2, B, G*G, 1)
    z_pres_similarity = (1 - sq_delta(z_press[2:], z_press[:-2]))
    z_pres_deltas = (sq_delta(z_press[2:], z_press[1:-1]) + sq_delta(z_press[:-2], z_press[1:-1]))
    z_pres_inconsistencies = z_pres_similarity * z_pres_deltas
    return torch.sum(z_pres_inconsistencies)
