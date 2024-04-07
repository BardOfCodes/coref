import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
from configs.siri import SIRIConfFactory


class EvalConfFactory(CNF):

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
        super(EvalConfFactory, self).__init__()
        config = SIRIConfFactory(starting_weights, name, language_name, 
                                 machine, tokenization, data_gen_mode, 
                                 n_float_bins, seed, debug, 
                                 target).config

        config.exp_mode = "final_eval"
        config.trainer.name = "SimpleEvaluator"

        config.data_loader.validation_shapes.mode = "TEST"

        config.siri.rewriters.CGRewriter.cache_config.eval_mode = True
        # UPDATE This to the path of the precomputed subexprs
        config.siri.rewriters.CGRewriter.cache_config.subexpr_load_path = "/home/aditya/projects/coref/models/stage_13_siri_pre_2/all_subexpr.pkl"

        # evaluation
        config.siri.rewriters.PORewriter.n_processes = 1
        config.trainer.validation_batch_size = 1
        config.data_loader.validation_shapes.batch_size = 1
        config.data_loader.validation_synth.batch_size = 1
        config.data_loader.search.batch_size = 1
        # config.plad.search_config.beam_size = 1
        config.inference = CN()
        # Change these options to change inference mode.
        config.inference.n_rewrites = 3
        config.inference.rewriters = ["PO", "CP", "CG"]
        # config.inference.rewriters = ["CG", "PO", "CP"]
        # config.inference.rewriters = ["CG"]

        self.config = config
