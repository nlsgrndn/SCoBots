import torch


def flatten(nested_list):
    return torch.cat([e for lst in nested_list for e in lst])

def retrieve_latent_repr_from_logs(logs):
    z_where, z_pres_prob, z_what = logs['z_where'], logs['z_pres_prob'], logs['z_what']
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_what = z_what.detach().cpu()
    z_pres = z_pres_prob > 0.5
    return z_where, z_pres, z_pres_prob, z_what
