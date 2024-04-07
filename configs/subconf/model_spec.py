import os
from wacky import CfgNode as CN


def ModelSpec(lang_conf, device, dropout=0.20):

    config = CN()
    config.name = "BaseVPINet"
    config.dropout = dropout
    config.device = device
    config.prog_seq_len = lang_conf.n_max_tokens
    if lang_conf.name in ["MCSG2D", "MCSG3D"]:
        config.prog_token_count = lang_conf.n_float_bins + \
            lang_conf.n_unique_cmds + lang_conf.n_integer_bins + 2
    else:
        config.prog_token_count = lang_conf.n_float_bins + lang_conf.n_unique_cmds + 2
    config.post_attn_sizes = [256, 512, 256, 128, config.prog_token_count]
    # config.post_attn_sizes = [256, 128, 64, config.prog_token_count]
    config.n_dims = lang_conf.n_dims
    config.cnn_first_stride = 1
    if lang_conf.n_dims == 3:
        config.visual_seq_len = 8
    else:
        config.visual_seq_len = 16
    config.num_dec_layers = 8
    config.num_heads = 16
    config.head_dim = 16
    config.hidden_dim = 256
    config.use_flash_attention = False  # True
    config.old_attn_mode = True  # True

    return config
