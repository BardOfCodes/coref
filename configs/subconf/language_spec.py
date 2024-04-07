from wacky import CfgNode as CN


def LangSpec(name, resolution, max_actions, n_float_bins, max_canvas_count, n_integer_bins, tokenization):

    config = CN()
    config.resolution = resolution
    config.name = name
    config.n_max_tokens = max_actions
    config.n_float_bins = n_float_bins
    config.n_integer_bins = n_integer_bins

    # For state machine
    config.max_canvas_count = max_canvas_count
    config.tokenization = tokenization

    # Level 1
    if name == "PCSG2D":
        config.n_dims = 2
        config.n_unique_cmds = 6
        config.param_count = 5
        # For the generator:
        config.has_macros = False
        config.has_htrans = False
        config.add_rotate = True
        config.convert_to_primal = True
    elif name == "PCSG3D":
        config.n_dims = 3
        config.n_unique_cmds = 5
        config.param_count = 6
        config.has_macros = False
        config.has_htrans = False
        # For the generator:
        config.add_rotate = False
        config.convert_to_primal = True
    # Level 2
    elif name == "HCSG2D":
        config.n_dims = 2
        config.n_unique_cmds = 9
        config.has_macros = False
        config.has_htrans = True
        # For the generator:
        config.add_rotate = True
        config.convert_to_primal = False
    elif name == "HCSG3D":
        config.n_dims = 3
        config.n_unique_cmds = 9
        config.has_macros = False
        config.has_htrans = True
        # For the generator:
        config.add_rotate = True
        config.convert_to_primal = False
    # Level 3
    elif name == "MCSG2D":
        config.n_dims = 2
        config.n_unique_cmds = 14
        config.has_macros = True
        config.has_htrans = True
        # For the generator:
        config.add_rotate = False
        config.convert_to_primal = False
    elif name == "MCSG3D":
        config.n_dims = 3
        config.n_unique_cmds = 18
        config.has_macros = True
        config.has_htrans = True
        # For the generator:
        config.add_rotate = False
        config.convert_to_primal = False
    return config
