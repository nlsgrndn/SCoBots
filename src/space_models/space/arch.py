from yacs.config import CfgNode

arch = CfgNode({
    # TR
    'adjacent_consistency_weight': 0.0,
    'pres_inconsistency_weight': 0.0,
    'area_pool_weight': 0.0,
    'area_object_weight': 10.0,
    'cosine_sim': True,
    'object_threshold': 0.5,
    'z_cos_match_weight': 5.0,
    'full_object_weight': 3000,  # Unused
    'motion_input': True,
    'motion': True,
    'motion_kind': 'mode',
    'motion_direct_weight': 1.0,  # Unused
    'motion_loss_weight_z_pres': 1000.0, #10.0
    'motion_loss_weight_z_where': 10000.0, #100.0
    'motion_loss_weight_alpha': 5, #100, #1
    'motion_weight': 100.0,
    'motion_sigmoid_steepen': 10000.0,  # Unused
    'motion_cooling_end_step': 3000,
    'motion_cooling_start_step': 0,
    'dynamic_scheduling': True,
    'agree_sim': True,
    'dynamic_steepness': 2.0,
    'use_variance': True,
    'motion_underestimating': 1.25, #2.0,
    'motion_object_found_lambda': 0.1, #0.025,
    'z_where_offset': 0.1,
    'acceptable_non_moving': 8,  # Unused
    'variance_steps': 20,
    'motion_requirement': 2.0,  # Unused
    # SPACE-config
    'img_shape': (128, 128),
    'T': 4,
    
    # Grid size. There will be G*G slots
    'G': 8,
    
    # Foreground configurations
    # ==== START ====
    # Foreground likelihood sigma
    'fg_sigma': 0.2, #0.15,
    # Size of the glimpse
    'glimpse_size': 32,
    # Encoded image feature channels
    'img_enc_dim_fg': 128,
    # Latent dimensions
    'z_pres_dim': 1,
    'z_depth_dim': 1,
    # (h, w)
    'z_where_scale_dim': 2,
    # (x, y)
    'z_where_shift_dim': 2,
    'z_what_dim': 32,
    
    # z_pres prior
    'z_pres_start_step': 0, #4000,
    'z_pres_end_step': 5000, #10000,
    'z_pres_start_value': 0.1,
    'z_pres_end_value': 1e-10, #0.01,
    
    # z_scale prior
    'z_scale_mean_start_step': 0, # 10000
    'z_scale_mean_end_step': 5000, # 20000
    'z_scale_mean_start_value': -2.0, # -1.0
    'z_scale_mean_end_value': -2.5, # -2.0
    'z_scale_std_value': 0.1,
    
    # Temperature for gumbel-softmax
    'tau_start_step': 0,
    'tau_end_step': 10000,
    'tau_start_value': 2.5,
    'tau_end_value': 2.5,
    
    # Fix alpha for the first N steps
    'fix_alpha_steps': 0,
    'fix_alpha_value': 0.1,
    # ==== END ====
    
    
    # Background configurations
    # ==== START ====
    # Number of background components. If you set this to one, you should use a strong decoder instead.
    'K': 3, # 5
    # Background likelihood sigma
    'bg_sigma': 0.1, #0.15,
    # Image encoding dimension
    'img_enc_dim_bg': 64,
    # Latent dimensions
    'z_mask_dim': 32,
    'z_comp_dim': 32,
    
    # (H, W)
    'rnn_mask_hidden_dim': 64,
    # This should be same as above
    'rnn_mask_prior_hidden_dim': 64,
    # Hidden layer dim for the network that computes q(z_c|z_m, x)
    'predict_comp_hidden_dim': 64,
    # ==== END ====
})