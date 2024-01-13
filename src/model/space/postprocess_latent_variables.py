import torch


def z_where_to_bb_format(width, height, center_x, center_y):
    center_x = (center_x + 1.0) / 2.0
    center_y = (center_y + 1.0) / 2.0
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2
    return y_min, y_max, x_min, x_max

def bb_to_z_where_format(y_min, y_max, x_min, x_max):
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    center_x = center_x * 2.0 - 1.0
    center_y = center_y * 2.0 - 1.0
    return width, height, center_x, center_y

def convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=False):
    """

    All inputs should be tensors

    :param z_where: (B, N, 4). [sx, sy, tx, ty]. N is arch.G ** 2
    :param z_pres: (B, N) Must be binary and byte tensor
    :param z_pres_prob: (B, N). In range (0, 1)
    :return: [[y_min, y_max, x_min, x_max, conf] * N] * B
    """
    B, N, _ = z_where.size()
    z_pres = z_pres.bool()

    # each (B, N, 1)
    width, height, center_x, center_y = torch.split(z_where, 1, dim=-1)
    y_min, y_max, x_min, x_max = z_where_to_bb_format(width, height, center_x, center_y)
    # (B, N, 4)
    pos = torch.cat([y_min, y_max, x_min, x_max], dim=-1)
    boxes_for_each_batch = []
    for b in range(B):
        # (N, 4), (N,) -> (M, 4), where M is the number of z_pres == 1
        boxes_pres = pos[b][z_pres[b]]
        # (N,) -> (M, 1)
        if with_conf:
            conf = z_pres_prob[b][z_pres[b]][:, None]
            # (M, 5)
            boxes_pres = torch.cat([boxes_pres, conf], dim=1)
        boxes_pres = boxes_pres.detach()
        boxes_pres = boxes_pres.cpu()
        boxes_pres = boxes_pres.numpy()
        boxes_for_each_batch.append(boxes_pres)
    return boxes_for_each_batch

def convert_to_boxes_gpu(z_where, z_pres, z_pres_prob, with_conf=False):
    """

    All inputs should be tensors

    :param z_where: (B, N, 4). [sx, sy, tx, ty]. N is arch.G ** 2
    :param z_pres: (B, N) Must be binary and byte tensor
    :param z_pres_prob: (B, N). In range (0, 1)
    :return: [[y_min, y_max, x_min, x_max, conf] * N] * B
    """
    B, N, _ = z_where.size()
    z_pres = z_pres.bool()

    # each (B, N, 1)
    width, height, center_x, center_y = torch.split(z_where, 1, dim=-1)
    y_min, y_max, x_min, x_max = z_where_to_bb_format(width, height, center_x, center_y)
    # (B, N, 4)
    pos = torch.cat([y_min, y_max, x_min, x_max], dim=-1)
    boxes_for_each_batch = []
    for b in range(B):
        # (N, 4), (N,) -> (M, 4), where M is the number of z_pres == 1
        boxes_pres = pos[b][z_pres[b]]
        # (N,) -> (M, 1)
        if with_conf:
            conf = z_pres_prob[b][z_pres[b]][:, None]
            # (M, 5)
            boxes_pres = torch.cat([boxes_pres, conf], dim=1)
        boxes_pres = boxes_pres.detach()
        boxes_for_each_batch.append(boxes_pres)
    return boxes_for_each_batch


def retrieve_latent_repr_from_logs(logs):
    z_where, z_pres_prob, z_what = logs['z_where'], logs['z_pres_prob'], logs['z_what']
    z_where = z_where.detach()
    z_pres_prob = z_pres_prob.detach().squeeze(-1)
    z_what = z_what.detach()
    z_pres = z_pres_prob > 0.5
    return z_where, z_pres, z_pres_prob, z_what

def latent_to_boxes_and_z_whats(latent_logs_dict):
    z_where, z_pres, z_pres_prob, z_what = retrieve_latent_repr_from_logs(latent_logs_dict)
    bboxes = convert_to_boxes_gpu(z_where, z_pres, z_pres_prob)
    z_whats = z_what[z_pres]
    return bboxes, z_whats