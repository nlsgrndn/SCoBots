import torch


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
    # import ipdb; ipdb.set_trace()

    # each (B, N, 1)
    width, height, center_x, center_y = torch.split(z_where, 1, dim=-1)

    center_x = (center_x + 1.0) / 2.0
    center_y = (center_y + 1.0) / 2.0
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2
    # (B, N, 4)
    pos = torch.cat([y_min, y_max, x_min, x_max], dim=-1)
    boxes = []
    for b in range(B):
        # (N, 4), (N,) -> (M, 4), where M is the number of z_pres == 1
        box = pos[b][z_pres[b]]
        # (N,) -> (M, 1)
        if with_conf:
            conf = z_pres_prob[b][z_pres[b]][:, None]
            # (M, 5)
            box = torch.cat([box, conf], dim=1)
        box = box.detach().cpu().numpy()
        boxes.append(box)

    return boxes


def retrieve_latent_repr_from_logs(logs):
    z_where, z_pres_prob, z_what = logs['z_where'], logs['z_pres_prob'], logs['z_what']
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze(-1)
    z_what = z_what.detach().cpu()
    z_pres = z_pres_prob > 0.5
    return z_where, z_pres, z_pres_prob, z_what