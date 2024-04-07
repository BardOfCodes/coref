import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
from configs.pretrain import PretrainConfFactory
from configs.subconf.data_loader_spec import PLADDataLoaderSpec


class PLADConfFactory(CNF):

    def __init__(self,
                 starting_weights=None,
                 name="Debug",
                 language_name="PCSG3D",
                 machine="local",
                 tokenization="PREFIX",
                 data_gen_mode="PRECOMPILED",
                 n_float_bins=33,
                 seed=0,
                 debug=False,
                 target="DEFAULT"):
        super(PLADConfFactory, self).__init__()
        config = PretrainConfFactory(name, language_name, machine, tokenization,
                                     data_gen_mode,
                                     n_float_bins, seed, debug).config
        # Just set the additionall settings
        lang_conf = config.language_conf
        config.exp_mode = "PLAD"
        config.trainer.name = f"{config.exp_mode}Trainer"
        # 500 iterations
        train_batch_size = config.data_loader.train.batch_size
        epoch_size = train_batch_size * 500
        config.data_loader = PLADDataLoaderSpec(config.data_dir, lang_conf,
                                                epoch_size,
                                                train_batch_size,
                                                config.trainer.validation_batch_size,
                                                target=target)
        config.eval_specs.train_val_beam_size = 3

        plad_config = CN()
        plad_config.inner_patience = 5
        plad_config.max_inner_iter = 7
        plad_config.outer_patience = 200
        plad_config.max_outer_iter = 50_000

        if starting_weights is None:
            if language_name == "PCSG3D":
                plad_config.starting_weights = "/home/aditya/projects/coref/models/pcsg3d_pre_fixed_old/model_400.pt"
            elif language_name == "PCSG2D":
                ...
            elif language_name == "MCSG3D":
                ...
            elif language_name == "MCSG2D":
                ...
        else:
            plad_config.starting_weights = starting_weights

        plad_config.search_config = CN()
        plad_config.search_config.beam_size = 10
        plad_config.search_config.stochastic = False
        # TODO: Adding parallel for beam search
        plad_config.search_config.n_processes = 4

        config.plad = plad_config

        bpds_config = CN()
        bpds_config.n_expr_per_entry = 1
        config.bpds = bpds_config

        config.model.dropout = 0.1

        ws_config = CN()
        ws_config.model = config.model.clone()
        ws_config.active = True
        ws_config.model.name = "VPINetVAE"
        ws_config.model.vae_latent_dim = 256
        ws_config.starting_weights = plad_config.starting_weights
        ws_config.inner_patience = 5
        ws_config.max_inner_iter = 7
        ws_config.kld_loss_weight = 1.0

        config.ws_config = ws_config

        ws_eval_specs = config.eval_specs.clone()
        ws_eval_specs.n_samples = 1_000
        ws_eval_specs.beam_size = 1

        config.ws_eval_specs = ws_eval_specs

        # Update config
        config.optimizer.kwargs.lr = 0.001

        self.config = config
