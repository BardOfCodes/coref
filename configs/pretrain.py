import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF

from configs.subconf.machine_spec import MachineSpec
from configs.subconf.language_spec import LangSpec
from configs.subconf.data_loader_spec import PretrainDataLoaderSpec
from configs.subconf.model_spec import ModelSpec
from configs.subconf.opt_spec import OptimizerSpec


class PretrainConfFactory(CNF):

    def __init__(self, name="Debug",
                 language_name="PCSG3D",
                 machine="local",
                 tokenization="PREFIX",
                 data_gen_mode="STREAMING",
                 n_float_bins=33,
                 seed=0,
                 debug=False):
        super(PretrainConfFactory, self).__init__()
        MAX_CANVAS_COUNT = 12
        PARSIMONY_FACTOR = 0.01 # 0.0001

        if language_name == "PCSG3D":
            MAX_ACTIONS = MAX_CANVAS_COUNT * 7 + (MAX_CANVAS_COUNT - 1) + 2 + 1
            RESOLUTION = 32
            TRAIN_BATCH_SIZE = 512
            VALIDATION_BATCH_SIZE = 256
            N_INTEGER_BINS = 0
        elif language_name == "PCSG2D":
            MAX_ACTIONS = MAX_CANVAS_COUNT * 6 + (MAX_CANVAS_COUNT - 1) + 2 + 1
            RESOLUTION = 64
            TRAIN_BATCH_SIZE = 512
            VALIDATION_BATCH_SIZE = 256
            N_INTEGER_BINS = 0
        elif language_name == "HCSG2D":
            MAX_ACTIONS = 128
            RESOLUTION = 64
            TRAIN_BATCH_SIZE = 512
            VALIDATION_BATCH_SIZE = 256
            N_INTEGER_BINS = 0
        elif language_name == "MCSG3D":
            MAX_ACTIONS = 128
            RESOLUTION = 32
            TRAIN_BATCH_SIZE = 512
            VALIDATION_BATCH_SIZE = 150
            N_INTEGER_BINS = 4
        elif language_name == "MCSG2D":
            MAX_ACTIONS = 128
            RESOLUTION = 64
            TRAIN_BATCH_SIZE = 512
            VALIDATION_BATCH_SIZE = 150
            N_INTEGER_BINS = 4

        config = CN()
        config.exp_mode = "Pretrain"
        config.trainer = CN()
        config.trainer.name = f"{config.exp_mode}Trainer"
        config.trainer.validation_batch_size = VALIDATION_BATCH_SIZE
        EPOCH_SIZE = TRAIN_BATCH_SIZE * 4000
        config.trainer.max_epochs = 5000
        config.trainer.score_tolerance = 0.001
        config.trainer.save_frequency = 25
        config.trainer.load_weights = ""
        config.trainer.entropy_loss_weight = 0.01

        config.dtype = "float32"
        config.device = "cuda"
        config.seed = seed
        config.resolution = RESOLUTION
        config.name = name

        config.machine_specs = MachineSpec(machine)
        config.project_dir = config.machine_specs.project_dir
        config.data_dir = config.machine_specs.data_dir
        config.model_save_dir = os.path.join(
            config.project_dir, "models", name)

        lang_conf = LangSpec(language_name, RESOLUTION, MAX_ACTIONS,
                             n_float_bins, MAX_CANVAS_COUNT, N_INTEGER_BINS, tokenization)
        config.language_conf = lang_conf
        config.data_loader = PretrainDataLoaderSpec(config.data_dir, lang_conf,
                                                    EPOCH_SIZE,
                                                    TRAIN_BATCH_SIZE,
                                                    VALIDATION_BATCH_SIZE,
                                                    data_gen_mode,
                                                    train_n_programs=EPOCH_SIZE,
                                                    validation_n_programs=1000)

        config.model = ModelSpec(lang_conf, device=config.device)

        optimizer, scheduler = OptimizerSpec()  # epoch is used for lr scheduler?
        config.optimizer = optimizer
        config.scheduler = scheduler

        config.logger = CN()
        config.logger.project_name = "coref"
        config.logger.user_name = "bardofcodes"
        config.logger.verbose = True
        config.logger.log_interval = 100
        config.logger.use_wandb = False  # not debug
        config.logger.logdir = os.path.join(
            config.project_dir, "logs", config.name)

        # validation here itself
        config.eval_specs = CN()
        config.eval_specs.parsimony_factor = PARSIMONY_FACTOR
        # TODO: DO speed ablation on this
        config.eval_specs.sdf_eval_bs = 256
        config.eval_specs.n_samples = 1000
        config.eval_specs.bs_eval_frequency = 1
        config.eval_specs.beam_size = 10

        config.notification = CN()
        config.notification.enable = False  # not debug
        config.notification.channel = "aditya"
        config.notification.webhook = "https://hooks.slack.com/services/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"

        # DEBUG:
        self.config = config
