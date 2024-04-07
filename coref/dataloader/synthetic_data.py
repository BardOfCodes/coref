from collections import defaultdict
import numpy as np
import torch as th
import os
import gc
from pathlib import Path
import _pickle as cPickle
import time
import coref.language as language
from geolipi.torch_compute.sketcher import Sketcher
from geolipi.torch_compute.compile_expression import create_compiled_expr
from coref.language.streaming_data_generator import generate_programs

TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
TEST = "TEST"

TRAIN_RATIO = 0.8  # amount of data used in training vs val for synthetic data.


class SynthDataset(th.utils.data.IterableDataset):

    def __init__(self, config, subset_config, lang_conf, device, dtype, seed, *args, **kwargs):

        # read data
        self.seed = seed
        np.random.seed(seed)

        self.bake_dir = config.bake_dir
        self.resolution = config.resolution
        self.epoch_size = config.epoch_size
        self.train_on_quantized = config.train_on_quantized
        self.total_programs_to_generate = subset_config.n_programs
        self.mode = subset_config.mode
        self.valid_prim_sizes = subset_config.valid_prim_sizes
        self.sampling_rate_per_size = subset_config.sampling_rate_per_size
        self.data_dir = subset_config.data_dir
        self.data_gen = subset_config.data_gen

        self.device = device
        self.dtype = dtype
        self.lang_conf = lang_conf
        self.epoch_count = 0

        rate_sum = np.sum(self.sampling_rate_per_size)
        self.sampling_rate_per_size = [
            x / rate_sum for x in self.sampling_rate_per_size]

        start_time = time.time()
        if self.data_gen.mode == "PRECOMPILED":
            print("Loading Cache...")
            cache = self.load_precompiled_data(
                device, dtype, create_cache=self.data_gen.cached_form)
        elif self.data_gen.mode == "STREAMING":
            print("Generating Cache...")
            cache = self.generate_streaming_data(
                create_cache=self.data_gen.cached_form)
        end_time = time.time()
        print(f"Cache loaded in {end_time - start_time} seconds.")
        self.cache_size = np.sum([len(x) for x in cache.values()])
        print(f"cache size is {self.cache_size}")
        self.cache = cache

        # now based on the "sampling rate, we have to create epoch entries"
        self.generate_data_keys()
        if self.mode == TRAIN:
            self.set_train_fetch_mode()
        else:
            self.set_val_fetch_mode()
        if not self.data_gen.cached_form:
            lang_specs = getattr(language, lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                           n_integer_bins=lang_conf.n_integer_bins,
                                                           tokenization=lang_conf.tokenization)
            self.lang_specs = lang_specs

    def quantize_cache(self, cache):
        lang_conf = self.lang_conf
        lang_specs = getattr(language, lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                       n_integer_bins=lang_conf.n_integer_bins,
                                                       tokenization=lang_conf.tokenization)
        revert_to_base_expr = lang_specs.revert_to_base_expr
        create_action_spec = lang_specs.create_action_spec
        action_mapper = lang_specs.create_action_mapper(device=self.device)
        for key, value in cache.items():
            for ind, expression in enumerate(value):
                action_specs = create_action_spec(
                    expression, lang_conf.n_max_tokens, device="cpu")
                actions = action_specs[0][:action_specs[2]
                                          ].cpu().numpy().tolist()
                quantized_program = lang_specs.actions_to_expr(
                    actions, action_mapper)
                cache[key][ind] = quantized_program

        return cache

    def generate_data_keys(self):
        data_key_sets = []
        for ind, rate in enumerate(self.sampling_rate_per_size):
            size = self.valid_prim_sizes[ind]
            n_programs = len(self.cache[size])
            n_req = int(np.round(self.epoch_size * rate))
            # sample n_req indices from 0 to n_programs

            if self.mode == TRAIN:
                if n_req > n_programs:
                    indices = np.random.choice(n_programs, n_req, replace=True)
                else:
                    indices = np.random.choice(
                        n_programs, n_req, replace=False)
            else:
                indices = [i % n_programs for i in range(n_req)]
            data_key_sets.append([(size, ind) for ind in indices])
        # interleave the data keys
        data_keys = []
        m = max([len(x) for x in data_key_sets])

        for i in range(m):
            for j in range(len(data_key_sets)):
                if i < len(data_key_sets[j]):
                    data_keys.append(data_key_sets[j][i])

        if self.mode == TRAIN:
            np.random.shuffle(data_keys)
        self.data_keys = data_keys

    def generate_streaming_data(self, create_cache):

        all_programs = dict()
        eval_sizes = self.data_gen.eval_sizes
        n_primitves = self.data_gen.n_primitives

        start_time = time.time()

        lang_conf = self.lang_conf
        sketcher = Sketcher(device=self.device, dtype=self.dtype,
                            resolution=self.resolution, n_dims=lang_conf.n_dims)
        lang_specs = getattr(language,  lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                        n_integer_bins=lang_conf.n_integer_bins,
                                                        tokenization=lang_conf.tokenization)
        for ind, rate in enumerate(self.sampling_rate_per_size):
            size = self.valid_prim_sizes[ind]
            n_req = int(np.round(self.total_programs_to_generate * rate))
            programs = generate_programs(size, n_req, sketcher, lang_conf, lang_specs,
                                         eval_size=eval_sizes[ind], n_primitive_set=n_primitves[ind])
            end_time = time.time()
            print(f"Generated programs in {end_time - start_time} seconds.")
            all_programs[size] = programs

        if self.train_on_quantized:
            all_programs = self.quantize_cache(all_programs)
        if create_cache:
            cache = self.create_cache(all_programs, sketcher, lang_specs)
        else:
            cache = all_programs
        return cache

    def load_precompiled_data(self, device, dtype, create_cache):
        # compile data

        lang_conf = self.lang_conf
        bake_name = f"{self.mode}_{lang_conf.tokenization}_{lang_conf.n_float_bins}_{lang_conf.n_integer_bins}_{lang_conf.n_max_tokens}_data.pkl"
        self.bake_file = os.path.join(self.bake_dir, bake_name)

        lang_specs = getattr(language,  lang_conf.name)(n_float_bins=lang_conf.n_float_bins,
                                                        n_integer_bins=lang_conf.n_integer_bins,
                                                        tokenization=lang_conf.tokenization)
        if create_cache:
            if os.path.exists(self.bake_file):
                print(f"Loading cache from {self.bake_file}")
                gc.disable()
                cache = cPickle.load(open(self.bake_file, "rb"))
                gc.enable()
            else:
                # create cache
                print("Cache not found. Creating cache.")
                all_programs = self.load_program_files(lang_specs)
                sketcher = Sketcher(device=device, dtype=dtype, resolution=self.resolution,
                                    n_dims=lang_conf.n_dims)
                if self.train_on_quantized:
                    all_programs = self.quantize_cache(all_programs)
                cache = self.create_cache(all_programs, sketcher, lang_specs)
                # check and create directory
                Path(self.bake_dir).mkdir(parents=True, exist_ok=True)
                print(f"Saving cache to {self.bake_file}")
                gc.disable()
                cPickle.dump(cache, open(self.bake_file, "wb"))
                gc.enable()
        else:
            all_programs = self.load_program_files(lang_specs)
            if self.train_on_quantized:
                all_programs = self.quantize_cache(all_programs)
            cache = all_programs
        return cache

    def load_program_files(self, lang_specs):
        # disable garbage collector
        all_programs = defaultdict(list)
        for k in self.valid_prim_sizes:
            print(f"Loading programs with {k} primitives")
            gc.disable()
            file_name = os.path.join(self.data_dir, f"{self.lang_conf.name}",
                                     f'{self.lang_conf.name}_synth_programs_{k}.pkl')
            with open(file_name, 'rb') as f:
                programs = cPickle.load(f)
            # enable garbage collector again
            gc.enable()
            train_limit = int(len(programs) * TRAIN_RATIO)
            if self.mode == TRAIN:
                programs = programs[:train_limit]
            elif self.mode == VALIDATION:
                programs = programs[train_limit:]
            programs = [eval(p, lang_specs.STR_TO_CMD_MAPPER)
                        for p in programs]
            all_programs[k] = programs
        return all_programs

    def create_cache(self, programs, sketcher, lang_specs):
        cache = defaultdict(list)
        revert_to_base_expr = lang_specs.revert_to_base_expr
        create_action_spec = lang_specs.create_action_spec
        # action_mapper = lang_specs.create_action_mapper(device=self.device)

        for size, programs in programs.items():
            print(f"Creating cache for programs with {size} primitives")
            for ind, program in enumerate(programs):
                if (ind+1) % 1000 == 0:
                    print(f"Creating cache for {ind + 1}/{len(programs)}")
                    # convert from language to base
                reverted_program = revert_to_base_expr(program)
                reverted_program = reverted_program.to(sketcher.device)
                cache_expr = create_compiled_expr(
                    reverted_program, sketcher=sketcher)
                # actions
                action_specs = create_action_spec(
                    program, self.lang_conf.n_max_tokens, device="cpu")
                # actions = action_specs[0][:action_specs[2]].cpu().numpy().tolist()
                # expression = lang_specs.actions_to_expr(actions, action_mapper)
                action_len = action_specs[-1]
                if action_len < self.lang_conf.n_max_tokens:
                    cache[size].append([cache_expr, action_specs])
        return cache

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        self.reset()
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem_train_cached(self, index):

        index = index % len(self.data_keys)
        size, index = self.data_keys[index]

        expr_obj, action_obj = self.cache[size][index]
        expr, transforms, inversions, params = expr_obj
        actions, action_validity, n_actions = action_obj

        return expr, transforms, inversions, params, actions, action_validity, n_actions

    def getitem_val_cached(self, index):

        index = index % len(self.data_keys)
        size, index = self.data_keys[index]

        expr_obj, action_obj = self.cache[size][index]
        expr, transforms, inversions, params = expr_obj

        return expr, transforms, inversions, params, (size, index)

    def getitem_train_uncached(self, index):

        index = index % len(self.data_keys)
        size, index = self.data_keys[index]
        program = self.cache[size][index]
        actions, action_validity, n_actions = self.lang_specs.create_action_spec(program,
                                                                                 self.lang_conf.n_max_tokens,
                                                                                 device="cpu")
        return program, actions, action_validity, n_actions

    def getitem_val_uncached(self, index):
        index = index % len(self.data_keys)
        size, index = self.data_keys[index]
        program = self.cache[size][index]

        return program, (size, index)

    def reset(self):
        if self.mode == TRAIN:
            self.generate_data_keys()
        if self.data_gen.mode == "STREAMING":
            if (self.epoch_count != 0) and (self.epoch_count % self.data_gen.regen_interval == 0):
                start_time = time.time()
                cache = self.generate_streaming_data(
                    create_cache=self.data_gen.cached_form)
                end_time = time.time()
                print(f"Cache loaded in {end_time - start_time} seconds.")
                self.cache_size = np.sum([len(x) for x in cache.values()])
                print(f"cache size is {self.cache_size}")
                self.cache = cache
        self.epoch_count += 1

    def set_train_fetch_mode(self):
        if self.data_gen.cached_form:
            self.getitem = self.getitem_train_cached
        else:
            self.getitem = self.getitem_train_uncached

    def set_val_fetch_mode(self):
        if self.data_gen.cached_form:
            self.getitem = self.getitem_val_cached
        else:
            self.getitem = self.getitem_val_uncached

    def get_target(self, key):

        key_bits = key.split("_")
        primary = "_".join(key_bits[:-1])
        index = int(key_bits[-1])
        target = self.cache[primary][index]
        return target
