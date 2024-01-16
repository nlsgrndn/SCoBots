
import torch
from model.space.arch import arch
import torch.nn as nn
from model.space.utils import spatial_transform
class WrappedSPACEforInference(nn.Module):
    def __init__(self, space):
        super().__init__()
        self.img_encoder = space.fg_module.img_encoder
        self.z_what_net = space.fg_module.z_what_net

    def forward(self, x):

        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        ### copied from img_encoder ###
        B = x.size(0)
        # (B, C, H, W)
        img_enc = self.img_encoder.enc(x[:, :3])
        # (B, E, G, G)
        lateral_enc = self.img_encoder.enc_lat(img_enc)
        # (B, 2E, G, G) -> (B, 128, H, W) / G, G
        cat_enc = self.img_encoder.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))
        cat_enc_z_pres = cat_enc


        def reshape(*args):
            """(B, D, G, G) -> (B, G*G, D)"""
            out = []
            for x in args:
                B, D, G, G = x.size()
                y = x.permute(0, 2, 3, 1).view(B, G * G, D)
                out.append(y)
            return out[0] if len(args) == 1 else out
        # Compute posteriors
        grid_width = 128 // arch.G
        # (B, 1, G, G)
        avg = nn.AvgPool2d(grid_width + 2, grid_width, padding=1, count_include_pad=False)
        if arch.motion_input:
            cat_enc_z_pres = torch.cat((cat_enc_z_pres, avg(x[:, 3:4])), dim=1)
        z_pres_original = self.img_encoder.z_pres_net(cat_enc_z_pres)
        z_pres_logits_pure = torch.tanh(z_pres_original)
        # (B, 1, G, G) - > (B, G*G, 1)
        z_pres_logits = reshape(
            z_pres_logits_pure)
        # 8.8 is only here to allow sigmoid(tanh(x)) to be around zero and one
        z_pres_logits = 8.8 * z_pres_logits

        # (B, 2, G, G)
        z_scale_mean, _ = self.img_encoder.z_scale_net(cat_enc).chunk(2, 1)
        # (B, 2, G, G) -> (B, G*G, 2)
        z_scale_mean = reshape(z_scale_mean)
        z_scale = z_scale_mean
        
        # (B, 2, G, G)
        z_shift_mean, _ = self.img_encoder.z_shift_net(cat_enc).chunk(2, 1)
        # (B, 2, G, G) -> (B, G*G, 2)
        z_shift_mean = reshape(z_shift_mean)
        z_shift = z_shift_mean

        # scale: unbounded to (0, 1), (B, G*G, 2)
        z_scale = z_scale.sigmoid()
        # offset: (2, G, G) -> (G*G, 2)
        offset = self.img_encoder.offset.permute(1, 2, 0).view(arch.G ** 2, 2)
        # (B, G*G, 2) and (G*G, 2)
        # where: (-1, 1)(local) -> add center points -> (0, 2) -> (-1, 1)
        z_shift = (2.0 / arch.G) * (offset + 0.5 + z_shift.tanh()) - 1

        # (B, G*G, 4)
        z_where_pure = torch.cat((z_scale, z_shift), dim=-1)
        z_where = z_where_pure



        ### copied from fg_module ### ("in between section")
        # (B, 3, H, W) -> (B*G*G, 3, H, W). Note we must use repeat_interleave instead of repeat
        x_repeat = torch.repeat_interleave(x, arch.G ** 2, dim=0)
        # (B*G*G, 3, H, W), where G is the grid size
        # Extract glimpse
        x_att = spatial_transform(x_repeat, z_where.view(B * arch.G ** 2, 4),
                                      (B * arch.G ** 2, 3, arch.glimpse_size, arch.glimpse_size))

        z_what_input = x_att
        ### copied from z_what_net ###
        z_what_input = self.z_what_net.enc_cnn(z_what_input)
        z_what_mean, _ = self.z_what_net.enc_what(z_what_input.flatten(start_dim=1)).chunk(2, -1)
        z_what = z_what_mean

        z_what = z_what.view(B, arch.G ** 2, arch.z_what_dim)

        result_dict = {
            "z_pres_prob": torch.sigmoid(z_pres_logits),
            "z_where": z_where,
            "z_where_pure": z_where_pure,
            "z_what": z_what, #TODO: here .unsqueeze(0) might be needed
        }
        return result_dict