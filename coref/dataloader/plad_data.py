
import numpy as np

from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.compile_expression import create_compiled_expr
import coref.language as language
from .shapes_data import GTShapesDataset
from .synthetic_data import TRAIN, VALIDATION, TEST
# Need to load the datafile


class PLADShapesDataset(GTShapesDataset):

    def __init__(self, config, subset_config, lang_conf, device, dtype, seed, *args, **kwargs):

        self.category_labels = subset_config.category_labels
        self.data_dir = subset_config.data_dir

        super(GTShapesDataset, self).__init__(config, subset_config,
                                              lang_conf, device, dtype, seed, *args, **kwargs)
        self.n_max_tokens = self.lang_conf.n_max_tokens
        self._getitem = self._getshapesonly
        self.original_epoch_size = self.epoch_size
        self.init_programs()

    def init_programs(self):
        self.program_size = len(self.cache)
        self.rand_index = list(range(self.program_size))
        self.gt_cache = {}
        self.expr_cache = {}
        # Empty program
        self.programs = {}
        actions = np.zeros((self.n_max_tokens), dtype=np.int32)
        action_validity = np.zeros((self.n_max_tokens), dtype=np.bool)
        n_actions = 0
        action_specs = (actions, action_validity, n_actions)
        for index in range(self.cache_size):
            data_key = self.data_keys[index]
            cat_name, ind = data_key
            self.gt_cache[index] = self.cache[cat_name][ind]
            self.programs[index] = (data_key, 0, action_specs)

    def set_programs(self, programs):
        self.program_size = len(programs)
        self.rand_index = list(range(self.program_size))
        self.gt_cache = {}
        self.expr_cache = {}  # create cache from the programs
        self.programs = {}

        lang_specs = getattr(language,  self.lang_conf.name)(n_float_bins=self.lang_conf.n_float_bins,
                                                             n_integer_bins=self.lang_conf.n_integer_bins,
                                                             tokenization=self.lang_conf.tokenization)
        revert_to_base_expr = lang_specs.revert_to_base_expr
        create_action_spec = lang_specs.create_action_spec
        sketcher = Sketcher(device=self.device, dtype=self.dtype, resolution=self.resolution,
                            n_dims=self.lang_conf.n_dims)
        for index, program_obj in enumerate(programs):
            key, program, prog_type, origin_type = program_obj
            if prog_type == 0:
                self.gt_cache[index] = self.cache[key[0]][key[1]]
            else:
                # compile:
                reverted_program = revert_to_base_expr(program)
                reverted_program = reverted_program.to(sketcher.device)
                cache_expr = create_compiled_expr(
                    reverted_program, sketcher=sketcher)
                self.expr_cache[index] = cache_expr
            action_specs = create_action_spec(
                program, self.lang_conf.n_max_tokens, device="cpu")
            self.programs[index] = (key, prog_type, action_specs, origin_type)

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for i in range(self.epoch_size):
            if self.mode == TRAIN and i % self.program_size == 0:
                np.random.shuffle(self.rand_index)
            yield self[i]

    def __getitem__(self, index):
        return self._getitem(index)

    def to_plad_train_mode(self):
        self._getitem = self._getpladdata
        self.rand_index = list(range(self.program_size))
        np.random.shuffle(self.rand_index)
        self.epoch_size = self.original_epoch_size
        self.mode = TRAIN

    def to_plad_train_val_mode(self):
        self._getitem = self._getpladdata
        data_keys = [x for x in range(
            self.program_size) if self.programs[x][1] == 0]
        self.rand_index = data_keys
        self.epoch_size = self.cache_size
        self.mode = TRAIN

    def to_shapes_mode(self):
        self._getitem = self._getshapesonly
        self.epoch_size = self.cache_size
        self.mode = VALIDATION

    def _getpladdata(self, index):
        index = index % len(self.rand_index)
        index = self.rand_index[index]
        cur_program = self.programs[index]
        # actually should still be equally on all entries instead of biasing towards shapes with more examples.

        key, program_type, action_obj, origin_type = cur_program
        if program_type == 0:
            # its actions with GT
            data = self.gt_cache[index]
            expr_obj = []
        else:
            data = []
            expr_obj = self.expr_cache[index]

        return expr_obj, data, action_obj

    def _getshapesonly(self, index):
        index = index % len(self.data_keys)
        key = self.data_keys[index]
        category, ind = key
        data = self.cache[category][ind]
        return data, key
