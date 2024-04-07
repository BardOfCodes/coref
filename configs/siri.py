import os
from wacky import CfgNode as CN,  CfgNodeFactory as CNF
from configs.plad import PLADConfFactory
from configs.subconf.data_loader_spec import PLADDataLoaderSpec


class SIRIConfFactory(CNF):

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
        super(SIRIConfFactory, self).__init__()
        config = PLADConfFactory(starting_weights, name, language_name, machine, tokenization,
                                 data_gen_mode,
                                 n_float_bins, seed, debug, target).config

        config.exp_mode = "SIRI"
        config.trainer.name = f"{config.exp_mode}Trainer"

        # BPDS updates
        config.bpds.n_expr_per_entry = 3

        rewriter_base = CN()
        rewriter_base.n_prog_per_item = 1
        rewriter_base.language_conf = config.language_conf.clone()
        rewriter_base.eval_specs = config.eval_specs.clone()

        po_conf = rewriter_base.clone()
        po_conf.sample_ratio = 0.5
        po_conf.valid_origin = ["BS", "CP", "CG"]
        po_conf.opt_step_size = 0.01
        po_conf.n_iters = 250
        po_conf.n_processes = 12
        po_conf.scale_factors_param = [3, 10]

        cp_conf = rewriter_base.clone()
        cp_conf.sample_ratio = 1.0
        cp_conf.valid_origin = ["BS", "DO", "CG"]
        cp_conf.ex_threshold = 0.001
        cp_conf.ex_diff_threshold = 0.005

        cg_conf = rewriter_base.clone()
        cg_conf.top_k = 15
        cg_conf.rewrite_limit = 10
        cg_conf.node_masking_req = 0.5
        cg_conf.add_dummy = True
        cg_conf.use_canonical = True
        cg_conf.sample_ratio = 0.25
        cg_conf.max_expression_len = 12 + 11 + 1
        cg_conf.valid_origin = ["BS", "DO", "CP", "CG"]

        cache_config = CN()
        cache_config.use_canonical = cg_conf.use_canonical
        cache_config.save_dir = config.model_save_dir
        cache_config.max_masking_rate = 0.90  # NOT USED RIGHT NOW
        cache_config.cache_size = 100_000
        cache_config.merge_nlist = 300
        cache_config.merge_nprobe = 25
        cache_config.merge_bit_distance = 1
        cache_config.search_nlist = 200
        cache_config.search_nprobe = 10
        cache_config.n_program_for_merge = 15000  # not used
        cache_config.n_program_to_sample = 10_000
        cache_config.cache_max_size = 100_000
        cache_config.eval_mode = False
        cache_config.subexpr_load_path = ""
        cg_conf.cache_config = cache_config.clone()

        rewriters = CN()
        rewriters.names = ["PORewriter", "CPRewriter", "CGRewriter"]
        rewriters.PORewriter = po_conf.clone()
        rewriters.CPRewriter = cp_conf.clone()
        rewriters.CGRewriter = cg_conf.clone()

        siri_conf = CN()
        siri_conf.rewriters = rewriters
        # Configure each rewriter.
        config.siri = siri_conf.clone()

        self.config = config
